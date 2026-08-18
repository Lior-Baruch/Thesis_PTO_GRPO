# Email draft — 2026-07-26

**To:** Doron, Moshe, Kfir
**Attach:** `results_snapshot_2026-07-26.pdf` (11 slides, in this folder) + `Exp1_ICLR2025/paper.pdf`
(the ICLR paper)

---

**Subject:** Exp3 results + asking for a joint meeting on where to take this

Hi Doron, Moshe and Kfir,

The main comparison in the current experiment is finished, so this feels like the right point to
decide where it goes before I start writing, both toward a publication and toward the thesis itself.

Both methods, PTO (preference-tree + DPO) and GRPO, have now run 10 training iterations under
matched settings and are fully scored on the whole evaluation battery, on 96 fixed patient personas.
The attached snapshot is numbers and figures, with the interpretation deliberately left out so we
can do that part together. It opens with a one-slide reminder of the ICLR paper and a slide on what
changed between that experiment, the follow-up, and this one. I've also attached the paper itself,
mainly for convenience.

The three factual takeaways, so the attachment isn't a cold read: both methods improve a lot over
the untrained 1B model; PTO ends higher than GRPO at the matched final iteration, and also higher
than GRPO's own best iteration; and in both methods the score gains come together with a measurable
rise in MI-inconsistent therapist behaviour, which is why the evaluation now reports several metrics
the reward never saw.

What I'd like us to decide:

**1. How to tell the story.** There are a few framings and I don't want to pick one alone:

- an MI-oriented story: what it takes to train a small model toward genuine MI quality, and what
  the questionnaires do and don't capture;
- a method-oriented story: preference-tree vs group-relative optimization when the reward is an
  expensive LLM judge, with the stability difference as the finding;
- a look-ahead story: the lever from the earlier paper, extended to both methods;

and combinations of these. The choice sets both the venue we aim at and the shape of the thesis
chapter, so it probably comes first.

**2. Whether to run more experiments, and which.** Options on the table, in no particular order:

- **Finish the look-ahead (K=5) arms.** This is the one comparison still open, and the one the ICLR
  paper actually rests on: there K=5 beat K=0, whereas in the current experiment the partial K=5
  data shows no difference yet. The arms were paused part-way on API cost, not because of any
  result. Resuming needs a small budget, which is worth a short conversation on its own.
- **A second judge model.** Right now one model grades everything, so we can't report inter-rater
  reliability. The pipeline for a second grader is built and ready to run.
- **A different model to train.** In particular, starting from an instruction-tuned base instead of
  the raw base model, which would change the starting point substantially and may change how much
  headroom either method has.
- **Anything you'd add.** I'd genuinely rather hear your suggestions than defend my own list.

**3. Scope.** What goes into a paper and what stays in the thesis, and whether the two are written
in parallel or one after the other.

When would suit you? I'll arrange my schedule around you; I'm free any day except August 4th and
5th. In person or on Zoom, whichever is easier. It would be best to have all four of us, but if we
can't find a time that works for everyone, I'd rather go ahead whenever Kfir is available.

Thanks,
Lior

---

## Notes for me (not part of the email)

- Attachment is results-only by design. No recommendations, no next-steps slide — those are the
  meeting.
- Budget deliberately mentioned as a constraint with no figure; bring the ~$300-to-date number and
  a per-arm estimate to the meeting.
- The last slide lists what's already on disk, in case someone asks what an extra analysis would
  cost (answer for most of it: nothing, it's computed).
- The K=5 slide states the tension with the ICLR result plainly (there K=5 won; here nothing yet on
  4 iterations of one method). Expect that to be the main question — the honest answer is that the
  current K comparison is underpowered, not negative.
- Doron and Moshe are already briefed (weekly meeting); Doron asked for the email. So the opening
  line is context for them and an entry point for Kfir — the real audience for the background
  slides is Kfir.
- Priority if scheduling gets hard: **meet whenever Kfir can**, even without Doron/Moshe. Stated in
  the email, so nobody has to guess.
- Only blocked dates: **August 4–5**.
