# iotj-revision-v12 — proposed paper revision release

Use the verified numerical attachment with `scripts/download_paper_inputs.py`, then run `scripts/reproduce_paper.py --output new_saved_output` and `scripts/reproduce_figures.py --assets-dir paper_revision/numerics/paper_assets --output new_figures`. The precise access status and archive identity are in `ACCESS.json`. At this candidate stage, external access has not been opened and the final distribution list awaits explicit author approval.

This version corresponds to *CoordiWorld: Penalty-Coupled Utility Learning for Structured Trajectory Evaluation*. It adds the C1 method and code map, seven complete source copies, frozen configurations, provenance bindings and the paper team's plotting implementation. It preserves historical methods and unchanged model bytes. It makes no claim about journal acceptance or a DOI.

The final-layout local CPU acceptance reproduced all five frozen comparisons with rtol=0 and atol=1e-10: C1 87.21473047728695, S1 84.85050801137633, difference 2.3642224659106126, 95% CI [1.3342334425852622, 3.6479198659883108]. The numerical attachment uses stable order-preserving aliases, without changing observations, grouping, missing denominators or time anchors. Plot coordinates and parameters retain the sole approved implementation; only route identifier text changes.

The existing synthetic model-integrity check and three component imports are separate from saved-result recalculation. Full original-environment inference, native scoring and training are not validated by these checks. Third-party raw assets remain subject to their providers' acquisition terms. No new license is assigned. The internal v12 review ZIP must not be uploaded as a Release attachment.
