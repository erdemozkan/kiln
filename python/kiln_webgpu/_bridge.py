"""The Node.js process that runs Kiln, and the requests sent to it."""
import atexit
import os
import platform
import re
import shutil
import subprocess
import threading

from ._codec import decode, encode, pack, unpack
from ._types import KilnError

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_NODE = (22, 12)


def find_node():
    """Node's path: $KILN_NODE, else the `node` on PATH."""
    node = os.environ.get("KILN_NODE") or shutil.which("node")
    if not node:
        raise KilnError("Kiln runs its models in Node.js, which was not found. Install Node %d.%d or newer (https://nodejs.org), or set KILN_NODE to its path." % MIN_NODE)
    return node


def node_version(node):
    try:
        text = subprocess.check_output([node, "--version"], stderr=subprocess.STDOUT).decode("utf-8", "replace")
    except (OSError, subprocess.CalledProcessError) as err:
        raise KilnError("%s did not start: %s" % (node, err))
    m = re.search(r"v(\d+)\.(\d+)", text)
    if not m:
        raise KilnError("%s --version printed %r" % (node, text.strip()))
    return int(m.group(1)), int(m.group(2))


def find_kiln():
    """Kiln's built index.js: $KILN_JS, else the copy in this package, else a repository checkout's build."""
    given = os.environ.get("KILN_JS")
    if given:
        return given
    for path in (os.path.join(HERE, "_js", "index.js"), os.path.join(HERE, "..", "..", "packages", "kiln-webgpu", "dist", "index.js")):
        if os.path.isfile(path):
            return os.path.abspath(path)
    raise KilnError("Kiln's index.js is missing from this install. In a checkout of the repository run `npm run build:python`, or set KILN_JS to the file.")


def gpu_home():
    """Where `install_gpu` puts Dawn's WebGPU package: $KILN_WEBGPU_HOME, else a per-user cache folder."""
    given = os.environ.get("KILN_WEBGPU_HOME")
    if given:
        return given
    if os.name == "nt":
        return os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "kiln-webgpu")
    return os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "kiln-webgpu")


def gpu_blocked(system=None, release=None):
    """Why the GPU is off in Node.js here, or None: on macOS before 15 Kiln runs on the CPU (see install_gpu).

    Read from the kernel's version (Darwin 24 is macOS 15): Python built for an older SDK reports
    macOS 11 and later as 10.16.
    """
    if system is None:
        system, release = platform.system(), platform.release()
    if system == "Darwin":
        try:
            if int(release.split(".")[0]) < 24:
                return "The GPU in Node.js needs macOS 15 or newer. Older versions of Dawn's webgpu package, the only ones that load on an older macOS, can hang the GPU; Kiln runs on the CPU instead."
        except ValueError:
            pass
    return None


def install_gpu(version=None):
    """Installs Dawn's WebGPU package for Node.js with npm, into this user's cache folder (about 140 MB).

    After it, `device="gpu"` and the default `device="auto"` run on the GPU. Returns the folder.
    """
    blocked = gpu_blocked()
    if blocked:
        raise KilnError(blocked)
    node = find_node()
    npm = shutil.which("npm") or shutil.which("npm", path=os.path.dirname(os.path.realpath(node)))
    if not npm:
        raise KilnError("npm was not found beside Node.js; install it, or run `npm install webgpu` in your project's folder yourself")
    home = gpu_home()
    os.makedirs(home, exist_ok=True)
    manifest = os.path.join(home, "package.json")
    if not os.path.isfile(manifest):
        with open(manifest, "w") as f:
            f.write('{"private": true}\n')
    spec = "webgpu@%s" % (version or "latest")
    try:
        subprocess.check_call([npm, "install", spec, "--no-audit", "--no-fund"], cwd=home)
    except (OSError, subprocess.CalledProcessError) as err:
        raise KilnError("npm install %s failed: %s" % (spec, err))
    BRIDGE.stop()
    return home


class Bridge:
    """One Node.js process for all pipelines, started at the first request."""

    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()
        self.next_id = 1

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        node = find_node()
        version = node_version(node)
        if version < MIN_NODE:
            raise KilnError("Kiln needs Node %d.%d or newer; %s is %d.%d" % (MIN_NODE + (node,) + version))
        env = dict(os.environ)
        # Kiln looks for Dawn's `webgpu` package from the working directory too; NODE_PATH adds
        # the folder install_gpu() fills.
        modules = os.path.join(gpu_home(), "node_modules")
        if os.path.isdir(modules):
            env["NODE_PATH"] = os.pathsep.join(filter(None, [env.get("NODE_PATH"), modules]))
        self.proc = subprocess.Popen([node, os.path.join(HERE, "_js", "bridge.mjs"), find_kiln()], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)

    def stop(self):
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass

    def events(self, op, body):
        """Sends a request and yields ("callback", function, args) as Kiln calls back, then ("result", value, None)."""
        if not self.lock.acquire(blocking=False):
            raise KilnError("Kiln is busy with another request (one at a time; a callback cannot start one)")
        done = False
        try:
            self.start()
            proc = self.proc
            blobs, callbacks = [], []
            rid = self.next_id
            self.next_id += 1
            frame = pack({"id": rid, "op": op, "body": encode(body, blobs, callbacks)}, blobs)
            try:
                proc.stdin.write(frame)
                proc.stdin.flush()
            except (OSError, ValueError):
                raise self.died()
            try:
                while True:
                    got = unpack(proc.stdout.read)
                    if got is None:
                        raise self.died()
                    header, received = got
                    if header.get("event") == "callback":
                        yield "callback", callbacks[header["callback"]], decode(header["args"], received)
                    elif header.get("id") == rid:
                        done = True
                        if not header.get("ok"):
                            raise KilnError(header["error"]["message"])
                        yield "result", decode(header.get("value"), received), None
                        return
            finally:
                # A caller that stops reading early (a stream it broke out of) leaves a request
                # running: its remaining messages are read here, so the next request starts clean.
                while not done and self.proc is proc:
                    got = unpack(proc.stdout.read)
                    if got is None:
                        self.stop()
                        break
                    done = got[0].get("id") == rid
        finally:
            self.lock.release()

    def died(self):
        code = self.proc.poll() if self.proc is not None else None
        self.stop()
        return KilnError("Node.js stopped%s; its messages are above" % ("" if code is None else " with exit code %d" % code))

    def request(self, op, body=None):
        """Sends a request, runs the callbacks it triggers, returns its result."""
        for kind, a, b in self.events(op, body or {}):
            if kind == "callback":
                a(*b)
            else:
                return a


BRIDGE = Bridge()
atexit.register(BRIDGE.stop)
