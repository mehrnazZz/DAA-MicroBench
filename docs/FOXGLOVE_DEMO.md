# Foxglove README Demo

This is the repeatable workflow for the README hero recording.

## Generate The Demo

```bash
python -m microbench.cli run \
  --scenario config/scenarios/urban_conflict_3d.yaml \
  --method reciprocal_velocity_obstacle \
  --n 4 \
  --seed 2 \
  --comm realistic_v2v_50hz \
  --out-dir runs_urban_conflict_demo

python -m microbench.cli foxglove-export \
  --trace runs_urban_conflict_demo/episodes/urban_conflict_3d_reciprocal_velocity_obstacle_n4_seed2_comm_realistic_v2v_50hz/trace_episode.jsonl \
  --out runs_urban_conflict_demo/urban_conflict_3d_rvo_avoidance.mcap \
  --trail-frames 2600 \
  --max-sensing-links 24 \
  --compression zstd
```

Open `runs_urban_conflict_demo/urban_conflict_3d_rvo_avoidance.mcap` in Foxglove Studio.

## Recommended Foxglove View

- Use the 3D panel as the main view.
- Show `/daa/static`, `/daa/agents`, `/daa/trails`, `/daa/intents`, and `/daa/sensing_links`.
- Leave `/daa/perception` off for the README recording unless you are explaining sensing range.
- Start near `t=10s`, when the conflict geometry becomes readable.
- Record about 15-20 seconds with the camera slightly above and offset from the crossing.

## Optimize The Recording

Save the raw recording outside the repo, then compress it into `docs/assets/`.

```bash
mkdir -p docs/assets
ffmpeg -i /path/to/raw_foxglove_recording.mov \
  -vf "scale=1280:-2,fps=30" \
  -c:v libx264 \
  -crf 28 \
  -preset slow \
  -pix_fmt yuv420p \
  -movflags +faststart \
  docs/assets/urban_conflict_3d_rvo_avoidance.mp4
```

Keep the committed asset small enough for a fast README load. If the MP4 is still too large, increase `-crf` to `30` or record a shorter clip.

## Embed At The Top Of README

After `docs/assets/urban_conflict_3d_rvo_avoidance.mp4` exists, add this near the top of `README.md` under the badges:

```html
<p align="center">
  <video src="docs/assets/urban_conflict_3d_rvo_avoidance.mp4" controls autoplay muted loop playsinline width="900"></video>
</p>
```

If GitHub does not render the relative MP4 in the README, upload the same MP4 to a GitHub release or issue comment and use the generated asset URL as the `src`.
