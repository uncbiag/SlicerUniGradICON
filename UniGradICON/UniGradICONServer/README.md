# UniGradICON registration server

Runs uniGradICON registrations for Slicers that cannot run them locally (for example, an Intel
macOS Slicer, whose PyTorch cannot use Slicer's NumPy 2) or not fast enough (no GPU). In the
UniGradICON module, check **Remote processing**, enter the server address (`hostname:port`) and,
if the server has one, the token, and press **Connect**.

If **Create transformed volume** is enabled in the client, the server creates it using the same ITK linear
resampling code as local registration and returns both the transform and the compressed image.
Otherwise it returns only the transform, skipping image resampling and download. The client
loads the returned image using Slicer's bundled SimpleITK and does not need local ITK or PyTorch.
The client's run button changes to **Cancel** for local and remote registrations. Transfers run
in the background, and status shows uploads, queue position, registration and downloads. A cancel
request during a transfer is sent once that transfer finishes; an active server job stops at its
next cancellation checkpoint. Cancelling a registration does not stop the registration server.

## Segmentations

The server accepts fixed and moving masks separately from Dice labels.
**Input masking** (`input_masking=true`) zeros intensities outside each mask before
registration and works without IO or Dice guidance. **Loss-function masking**
(`loss_function_masking=true`, `input_masking=false`) preserves image inputs and
restricts the similarity loss to the masks during IO. Both modes require
`fixed_mask` and `moving_mask` uploads; loss-function masking also requires positive
IO steps. Legacy fixed-mask-only requests retain their earlier behavior.

The server requirements install **uniGradICON 1.0.6** and
`icon_registration>=1.1.9`, including Dice guidance and masked similarity losses.
To update an existing server environment, run `pip install "unigradicon==1.0.6"`
and restart the server. No branch checkout is needed.

`/info` advertises `inputMasking` and adds `lossFunctionMasking` and
`segmentationGuidance` and `multiLabelSegmentation` when their installed APIs are available. Dice-guided jobs
require a positive `dice_loss_weight`, positive `io_steps`,
and both `fixed_segmentation` and `moving_segmentation` uploads. These are multi-label
images in physical coordinates, with matching nonzero integer label values. Label
values are preserved through nearest-neighbor regridding; all shared foreground
classes contribute to Dice. Clients set `multilabel_segmentation=true` and require
the advertised `multiLabelSegmentation` feature. Older servers or backends are
rejected before uploading instead of silently merging labels.
Segmentation propagation runs on the Slicer client using the returned transform
and nearest-neighbor interpolation; propagation inputs are not uploaded.

## Start it from Slicer

On a computer where UniGradICON runs locally, open **Local registration server**, choose the port,
enter a token so that only clients that know it can use the server, and press **Start server**.
The server address (`hostname:port`) is shown below the button; **Copy** puts it on the
clipboard, ready for the Server field of a client. **Log to GUI** shows the messages of the server
in the module's log, **Log to console** in the Python console. While the server is running, the
button changes to **Stop server**; click it to stop the server. To test from the same computer,
connect to `127.0.0.1:<port>`.

Other computers can only connect if the port is open in the firewall of the server computer. A
"Connection refused" from other computers while `curl http://localhost:<port>/info` works on the
server computer itself means the firewall blocks it, e.g. on RHEL/Rocky:
`sudo firewall-cmd --add-port=<port>/tcp --permanent && sudo firewall-cmd --reload`. Without
admin rights, an SSH tunnel works too: `ssh -N -L <port>:localhost:<port> user@server`, then
connect to `localhost:<port>`.

## Start it without Slicer

```bash
python -m venv unigradicon-server && source unigradicon-server/bin/activate
pip install torch                      # the build for your GPU, see https://pytorch.org
pip install -r requirements.txt
UNIGRADICON_SERVER_TOKEN=<secret> python server.py --host 0.0.0.0 --port 8899
```

Run it from a checkout of this repository: it imports `../UniGradICONHelpers`. The model weights
are read from `--weights-dir` (default: the repository's `model_checkpoints`) and downloaded there
on first use if missing.

## Notes

- Uploads must be self-contained `.nrrd`, `.nii.gz`, `.nii` or `.mha` files. Detached
  `.nhdr`/`.mhd` headers are rejected because their companion data files are not transferred;
  load the image in Slicer and save it as `.nrrd` first. Slicer's remote registration already
  exports self-contained NRRD files.
- Updated clients assign each submission a UUID (`request_id`) and reuse it on retries, so
  a lost upload response does not enqueue a second registration on an updated server. API
  callers should also reuse the same UUID when retrying a submission and generate a new one
  for each new registration. Older clients without this field are still accepted.
- The images are sent to the server. Use a token, and outside a trusted network put the server
  behind HTTPS (for example a reverse proxy); the server itself speaks plain HTTP.
- Jobs run in the order they arrive: one at a time on the CPU, and **Jobs per GPU**
  (`--jobs-per-gpu`, default 1) at a time on each GPU. Each of those keeps its own copy of the
  model in GPU memory and needs memory for its images on top, so try a higher number only with
  GPU memory to spare; jobs that run out of it fail. On one GPU, jobs at the same time share its
  compute, so the gain is mostly in overlapping transfers, loading and resampling. Finished jobs
  and their files are removed after the client downloads the results, or after an hour.
  A background cleanup checks every 30 seconds, including when the server is idle.
- `pytest test_server.py` tests the server with a fake registration (needs `httpx` and `pytest`).
  From this directory, `pytest test_server.py ../Testing/Python/test_registration.py ../Testing/Python/test_task.py` also
  checks concurrent downloads, optimization state restoration, background transfers and worker
  cancellation. Task tests use SimpleITK (bundled with Slicer). Optimization tests need
  the registration backend packages; no model weights or GPU are required for these tests.
