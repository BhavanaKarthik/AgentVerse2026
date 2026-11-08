from pathlib import Path

import modal

app = modal.App("isaac-lab-smoke")

ISAACLAB_REF = "v2.3.0"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .env({"DEBIAN_FRONTEND": "noninteractive", "ACCEPT_EULA": "Y", "OMNI_KIT_ACCEPT_EULA": "YES"})
    .apt_install(
        "git", "build-essential", "cmake", "libglu1-mesa", "libgl1", "libegl1", "libxext6",
        "libx11-6", "libxrandr2", "libxinerama1", "libxcursor1", "libxi6", "libsm6",
        "libice6", "libxt6", "libvulkan1", "vulkan-tools", "libglib2.0-0", "libxkbcommon0",
        "ffmpeg",
    )
    .pip_install("torch==2.7.0", "torchvision==0.22.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("isaacsim[all,extscache]==5.1.0", extra_index_url="https://pypi.nvidia.com")
    .run_commands(
        f"git clone --depth 1 --branch {ISAACLAB_REF} https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab",
        "pip install 'setuptools<70' wheel && pip install --no-build-isolation flatdict==4.0.1",
        "cd /opt/IsaacLab/source && ls && pip install -e isaaclab -e isaaclab_assets -e isaaclab_tasks"
        " --extra-index-url https://pypi.nvidia.com",
        "pip install click==8.1.7 idna==3.10 h5py && pip show isaaclab isaaclab_tasks",
    )
)
image = image.add_local_file("smoke.py", "/root/smoke.py")


@app.function(image=image, gpu="L40S", timeout=3000)
def smoke(task: str, eye: list, target: list) -> bytes:
    import subprocess

    out = "/tmp/out.png"
    cmd = ["python", "-u", "/root/smoke.py", "--task", task, "--out", out,
           "--eye", *map(str, eye), "--target", *map(str, target)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd="/opt/IsaacLab")
    print(r.stderr[-4000:])
    print("RETURNCODE", r.returncode)
    print("STDOUT_TAIL", r.stdout[-3000:])
    if "SMOKE_OK" not in r.stdout:
        raise RuntimeError("smoke test failed")
    return open(out, "rb").read()

@app.function(secrets=[modal.Secret.from_name("ngc-secret")])                                                           
def some_function():                                                                                                    
    os.getenv("NGC_API_KEY")    


@app.local_entrypoint()
def main(which: str = "franka"):
    cfgs = {
        "anymal": ("Isaac-Velocity-Flat-Anymal-D-v0", [-3.0, 0.0, 1.8], [0.0, 0.0, 0.4], "anymal_d_flat_third_person.png"),
        "franka": ("Isaac-Stack-Cube-Franka-v0", [1.4, -1.0, 1.1], [0.5, 0.0, 0.1], "stack_cube_franka.png"),
    }
    task, eye, target, fname = cfgs[which]
    png = smoke.remote(task, eye, target)
    path = Path(__file__).parent / "img" / fname
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(png)
    print("saved", path, len(png), "bytes")
