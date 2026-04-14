# Old DBSCAN GT vs. new hybrid AIS-anchored GT

**Pipeline:** old uses polar DBSCAN + fixed 12 px padding → uniform 33x33 boxes.
New hybrid: DBSCAN the polar points into per-target seeds, threshold the red channel, apply 5x5 closing, run connected components, greedy-match each AIS seed to its nearest bright blob (≤ 30 px), emit the blob's tight bbox.

**Colours:** garnet = old GT • atlantic = new matched GT • horseshoe (thin) = new unclaimed blobs (bright targets that did not match any AIS seed — likely non-cooperative vessels or strong clutter).

Files per frame:
- `<frame>_compare.png` – side-by-side 3-panel (old | new | overlay)
- `<frame>_1_old_gt.png` – old GT alone
- `<frame>_2_new_gt.png` – new GT (matched + unclaimed)
- `<frame>_3_overlay.png` – old + new on the raw capture
