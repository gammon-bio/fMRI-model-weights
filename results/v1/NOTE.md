# v1 results (superseded -- historical record)

Original Phase 1 spec, before the diagnosis task, the hard condition, and the review fixes.
The v1 gate reported GO, but an independent review showed two criteria were too weak:

- G2 (network recovery vs shuffled-label null): an *untrained* model also passes (6/6 seeds), because
  the masked-query attention readout inherits input FC at random init.
- G1 (beats pooled FC): most of the advantage is site-specific covariance plus temporal context;
  a per-site FC baseline closes 56-71% of the gap.

Do not cite these gate outcomes. See results/ (v2) for the current gate.
Code version: git history up to the commit that added this note.
