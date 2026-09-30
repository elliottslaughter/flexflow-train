#!/usr/bin/env python3
"""Check that the compiler fused every batch norm the runtime fusion pass would.

Reads the computation graph handed to compile-model and the mapped PCG it
wrote, finds every batch norm + activation pair in the former that
perform_operation_fusion (lib/task-spec/.../operation_fusion.cc) would fold
into one operator at execution time, and checks that each one arrives in the
latter already folded: the batch norm carrying the activation, and the
activation operator gone.

Batch norms are identified by the name of their GAMMA weight
("model.0.bn.GAMMA" belongs to "model.0.bn"). The weights keep their names
through the compiler's substitutions; a fused batch norm itself does not.

Prints PASSED or FAILED and exits non-zero on failure.
"""

import argparse
import collections
import json
import sys

# The pairs perform_operation_fusion folds, i.e. what
# batch_norm_supports_fused_activation and
# batch_norm_mode_supports_fused_activation allow.
FUSABLE_ACTIVATIONS = {"RELU", "SIGMOID", "TANH", "SILU"}
FUSABLE_MODES = {"SPATIAL", "SPATIAL_PERSISTENT"}


class Graph:
    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)["raw_graph"]
        self.labels = {node: label for node, label in raw["node_labels"]}
        self.inputs = collections.defaultdict(dict)  # node -> slot -> producer
        self.readers = collections.defaultdict(list)  # (node, slot) -> readers
        for e in raw["graph"]["edges"]:
            self.inputs[e["dstNode"]][e["dstSlot"]] = e["srcNode"]
            self.readers[(e["srcNode"], e["srcSlot"])].append(e["dstNode"])

    def op(self, node):
        attrs = self.labels[node]["op_attrs"]
        return attrs["type"], attrs["value"]

    def batch_norm_name(self, node):
        gamma = self.labels[self.inputs[node]["GAMMA"]].get("name") or ""
        assert gamma.endswith(".GAMMA"), (node, gamma)
        return gamma[: -len(".GAMMA")]

    def batch_norms(self):
        """Every batch norm, by name, with its attrs."""
        return {
            self.batch_norm_name(n): (n, self.op(n)[1])
            for n in self.labels
            if self.op(n)[0] == "batch_norm"
        }

    def op_type_counts(self):
        def key(n):
            t, v = self.op(n)
            return f"{t}:{v['op_type']}" if t == "element_unary" else t

        return collections.Counter(key(n) for n in self.labels)


def fusable_activation(graph, bn_node, bn_attrs):
    """The activation perform_operation_fusion would fold into this batch
    norm, or None."""
    if bn_attrs["activation"] is not None or bn_attrs["mode"] not in FUSABLE_MODES:
        return None
    readers = graph.readers[(bn_node, "OUTPUT")]
    if len(readers) != 1:
        return None
    kind, attrs = graph.op(readers[0])
    if kind != "element_unary" or attrs["op_type"] not in FUSABLE_ACTIVATIONS:
        return None
    if attrs["op_type"] == "SILU" and attrs.get("scalar") not in (None, 1.0):
        return None
    return attrs["op_type"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cg", required=True, help="compile-model's input")
    parser.add_argument("--mpcg", required=True, help="compile-model's output")
    parser.add_argument(
        "--expected-silu",
        type=int,
        help="how many batch norm + SiLU pairs the model is known to have",
    )
    args = parser.parse_args()

    before = Graph(args.cg)
    after = Graph(args.mpcg)

    # name -> activation that should be fused into it
    expected = {}
    for name, (node, attrs) in before.batch_norms().items():
        activation = fusable_activation(before, node, attrs)
        if activation is not None:
            expected[name] = activation

    by_activation = collections.Counter(expected.values())
    print(f"fusable batch norms in {args.cg}: {len(expected)}", dict(by_activation))

    failures = []
    if args.expected_silu is not None and by_activation["SILU"] != args.expected_silu:
        failures.append(
            f"the model has {by_activation['SILU']} batch norm + SiLU pairs, "
            f"not the {args.expected_silu} expected"
        )

    after_bns = after.batch_norms()
    missing = sorted(set(before.batch_norms()) - set(after_bns))
    if missing:
        failures.append(f"batch norms missing from the output: {missing}")

    not_fused = []
    for name, (node, attrs) in sorted(after_bns.items()):
        want = expected.get(name)
        if attrs["activation"] != want:
            not_fused.append(f"{name}: activation {attrs['activation']}, expected {want}")
    if not_fused:
        failures.append(
            f"{len(not_fused)} batch norms not fused as expected:\n    "
            + "\n    ".join(not_fused)
        )

    # Folding a pair removes exactly its activation operator. Anything else that
    # changed is some other substitution, which is reported but not judged.
    expected_counts = before.op_type_counts()
    for activation, n in by_activation.items():
        expected_counts[f"element_unary:{activation}"] -= n
    actual_counts = after.op_type_counts()
    for op in sorted(set(expected_counts) | set(actual_counts)):
        if expected_counts[op] != actual_counts[op]:
            print(
                f"note: {op}: {actual_counts[op]} in the output, "
                f"{expected_counts[op]} if only the fusions had been applied"
            )

    fused = sum(1 for name in expected if after_bns.get(name, (None, {}))[1].get("activation") == expected[name])
    print(f"fused in {args.mpcg}: {fused} of {len(expected)}")

    if failures:
        for f in failures:
            print(f"error: {f}")
        print("FAILED")
        return 1
    print(f"PASSED (all {len(expected)} fusable batch norms fused)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
