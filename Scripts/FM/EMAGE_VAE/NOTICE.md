# EMAGE AESK evaluator assets

This directory vendors the released EMAGE/PantoMatrix AESK evaluator used by
the official full-clip FGD protocol. The source was carried through MannerFlow
and is licensed under Apache License 2.0; see `LICENSE`.

StreamTalk changes are limited to an embedded, fixed SMPL-X 55-joint parent
table (the encoder only needs topology) and a fail-closed SHA-256 check for the
released `AESKConv_240_100.bin` checkpoint.

Expected checkpoint SHA-256:
`5cd9566b24264f34d44003b3de62cdfd50aa85b7cdde2d369214599023c40f55`.
