#!/usr/bin/env bash
# Pull "<tag>/<bench><TAB><accuracy>" for the v3 step-100 comparison off the eval box.
: "${EVAL_SSH:?set EVAL_SSH to the ssh command, e.g. \"ssh -i ~/.ssh/vast_ai -p 20113 root@host\"}"
ROOT="${OOD_EVAL_ROOT:-/workspace/runs/ood-eval-4b}"
$EVAL_SSH "for t in base commeff100 dense100; do
  for b in math500 gsm8k minerva olympiad mmlu_stem amc23 aime24 aime25 aime26 hmmt25; do
    f=$ROOT/\$t/\$b/train.log
    [ -f \"\$f\" ] || continue
    a=\$(grep -ao 'acc/mean@[0-9]*:[0-9.]*' \"\$f\" 2>/dev/null | tail -1 | sed 's/.*://')
    [ -n \"\$a\" ] && printf '%s/%s\t%s\n' \"\$t\" \"\$b\" \"\$a\"
  done
done" 2>/dev/null
