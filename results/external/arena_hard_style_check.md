# Default setup scored exactly Arena-Hard's way (19 October 2026)

Method: Arena-Hard's own `show_result.py` (commit 196f6b8), weighted wins with Bradley-Terry and style control
(length + markdown), fitted jointly over Arena-Hard's published GPT-4.1 judgments of all 27 models (Hugging Face
lmarena-ai/arena-hard-auto, data/arena-hard-v2.0), with gpt-4.1-mini's judgments replaced by ours
(`analysis/arena_hard_style_check_prepare.py`). Intervals are Arena-Hard's: 100 unseeded bootstrap draws,
5th-95th percentile (90%). Because the bootstrap is unseeded, scores move by a few tenths between runs of the script.

| Judgments for GPT-4.1 mini | Style control (Arena-Hard's leaderboard figure) | Weighted wins, no style control |
|---|---|---|
| Arena-Hard's own (published: 48.6) | 48.4 (-2.0 / +2.3); 48.7 on a second run of the script | 54.0 |
| Ours, May default setup | 48.0 (-1.7 / +2.3) | 54.0 |
| Ours, October run 1 | 47.6 (-2.6 / +2.6) | 53.2 |
| Ours, October run 2 | 50.2 (-2.2 / +2.6) | 55.6 |
| Ours, October run 3 | 47.9 (-2.3 / +2.3) | 53.8 |

GPT-4.1 mini's leaderboard position among the 28 models: 9th with Arena-Hard's judgments and with our May run;
10th, 6th and 10th with our three October runs (several neighbouring models sit within about 2 points).
