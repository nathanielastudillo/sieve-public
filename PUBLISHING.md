# About this snapshot

This repository is a **sanitised public snapshot** of a private working repository
(`sieve`). It is not the development history: it is a single commit of the tree as it
stood on 2026-10-13, prepared for public reading.

What was changed to make it publishable:

- **Infrastructure details replaced with placeholders.** Machine names, IP addresses,
  subnets, SSH users and absolute home paths appear as `<node-ip>`, `<lan>`,
  `<fabric-ip>`, `<standalone>`, `gpu-01`, `node-N`, `corpus-host`, `/home/<user>/`.
  The scripts are readable as an account of how the system was run; they are not
  runnable against the original fleet as written.
- **Password-based SSH removed.** The private copy carried a hardcoded fleet password
  and `sshpass` invocations; neither appears here.
- **Corpus provenance metadata withheld.** The private copy records where each training
  book was obtained. The corpus itself is not distributed, and the acquisition-source
  lists are not published.
- **Internal audit material excluded**, as were the personal documents that the private
  copy's data pipeline had accumulated.

What was *not* changed: the code, the lab notes (including the negative results), the
documentation, and the architecture descriptions. The negative results are deliberate —
`lab-notes/findings.md` records the small-corpus track being beaten by a unigram model
and the open-weight track not yet having run. Read that before judging the results.

The licence covers the code in this repository only, not any corpus material, model
weights, or third-party texts referenced in the notes.
