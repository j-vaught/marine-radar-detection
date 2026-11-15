"""One-use pinned TLS transport between explicitly configured fleet hosts."""

from __future__ import annotations

import ipaddress
import json
import secrets
import subprocess
import tempfile
import time
from pathlib import Path

from .fleet import quote_cmd, ssh_options

ALLOWED_SOURCES = {"comech-2422", "comech-1176", "cocesr-1034", "cocesr-1141"}

# The receiver creates an ephemeral certificate locally. The token is supplied via
# authenticated SSH stdin, and never appears in command arguments or logs.
RECEIVER = r"""
import hashlib,hmac,json,os,pathlib,socket,ssl,struct,subprocess,sys,tempfile,time
secret=bytes.fromhex(sys.stdin.readline().strip())
bind,expected_peer,stage,program=sys.argv[1:]
with tempfile.TemporaryDirectory(prefix='revision-tls-') as temp:
 os.chmod(temp,0o700)
 key=temp+'/key';cert=temp+'/cert'
 subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-keyout',key,'-out',cert,'-days','1','-subj','/CN=archive-ephemeral'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
 context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.load_cert_chain(cert,key)
 fingerprint=hashlib.sha256(ssl.PEM_cert_to_DER_cert(pathlib.Path(cert).read_text())).hexdigest()
 with socket.socket() as listener:
  listener.settimeout(60);listener.bind((bind,0));listener.listen(1)
  print(json.dumps({'port':listener.getsockname()[1],'fingerprint':fingerprint}),flush=True)
  conn,peer=listener.accept()
  if peer[0]!=expected_peer: raise ValueError('unexpected source address')
  conn.settimeout(60)
  with context.wrap_socket(conn,server_side=True) as tls:
   token=b''
   while len(token)<32:
    block=tls.recv(32-len(token))
    if not block: raise ValueError('missing token')
    token+=block
   if not hmac.compare_digest(token,secret): raise ValueError('invalid token')
   tls.settimeout(None)
   extractor=subprocess.Popen(['python3','-c',program,'extract',stage],stdin=subprocess.PIPE)
   count=0;start=time.monotonic()
   try:
    def exact(n):
     result=bytearray()
     while len(result)<n:
      block=tls.recv(n-len(result))
      if not block: raise RuntimeError('incomplete TLS frame')
      result.extend(block)
     return bytes(result)
    while True:
     size=struct.unpack('!I',exact(4))[0]
     if not size: break
     if size>1024*1024: raise ValueError('oversized TLS frame')
     block=exact(size);extractor.stdin.write(block);count+=len(block)
    extractor.stdin.close()
    if extractor.wait()!=0: raise RuntimeError('archive extraction failed after %d bytes' % count)
    tls.sendall(b'OK')
   except BaseException:
    extractor.kill();extractor.wait();raise
   print(json.dumps({'bytes':count,'seconds':time.monotonic()-start}),flush=True)
"""

SENDER = r"""
import hashlib,hmac,json,socket,ssl,struct,subprocess,sys,time
secret=bytes.fromhex(sys.stdin.readline().strip())
address,port,pin,source,program=sys.argv[1:]
ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT);ctx.check_hostname=False;ctx.verify_mode=ssl.CERT_NONE
with socket.create_connection((address,int(port)),timeout=15) as sock:
 with ctx.wrap_socket(sock,server_hostname='archive-ephemeral') as tls:
  if not hmac.compare_digest(hashlib.sha256(tls.getpeercert(binary_form=True)).hexdigest(),pin): raise ValueError('certificate pin mismatch')
  tls.sendall(secret);tls.settimeout(None)
  producer=subprocess.Popen(['python3','-c',program,'stream',source],stdout=subprocess.PIPE)
  count=0;start=time.monotonic()
  try:
   while True:
    block=producer.stdout.read(1024*1024)
    if not block: break
    tls.sendall(struct.pack('!I',len(block)));tls.sendall(block);count+=len(block)
   if producer.wait()!=0: raise RuntimeError('archive production failed')
   tls.sendall(struct.pack('!I',0))
   if tls.recv(2)!=b'OK': raise RuntimeError('receiver did not acknowledge extraction')
  except BaseException:
   producer.kill();producer.wait();raise
  print(json.dumps({'bytes':count,'seconds':time.monotonic()-start}),flush=True)
"""


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    bind = ipaddress.ip_address(config["destination_bind"])
    if bind.version != 4 or bind.is_unspecified or bind.is_loopback:
        raise ValueError("destination bind must be a specific IPv4 address")
    sources = config["sources"]
    if not isinstance(sources, dict) or not set(sources).issubset(ALLOWED_SOURCES):
        raise ValueError("transport config includes an unapproved source")
    for address in sources.values():
        ip = ipaddress.ip_address(address)
        if ip.version != 4 or ip.is_unspecified or ip.is_loopback:
            raise ValueError("source must be a specific IPv4 address")
    return config


def direct_copy(host: str, source: str, dest_host: str, stage: str, config: dict, program: str):
    if dest_host.lower() != "comech-2080" or host not in config["sources"]:
        raise ValueError("host not configured for direct transport")
    token = secrets.token_hex(32) + "\n"
    bind = config["destination_bind"]
    peer = config["sources"][host]
    receiver_cmd = quote_cmd(["python3", "-c", RECEIVER, bind, peer, stage, program])
    with tempfile.TemporaryFile() as receiver_err, tempfile.TemporaryFile() as sender_err:
        receiver = subprocess.Popen(
            ["ssh", *ssh_options(dest_host), dest_host, receiver_cmd],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=receiver_err,
            text=True,
        )
        sender = None
        output = ""
        try:
            assert receiver.stdin is not None and receiver.stdout is not None
            receiver.stdin.write(token)
            receiver.stdin.close()
            receiver.stdin = None
            # Listener has a 60-second accept timeout. Certificate creation and SSH
            # startup are bounded separately to prevent an indefinitely hung client.
            import select

            ready, _, _ = select.select([receiver.stdout], [], [], 60)
            if not ready:
                raise TimeoutError("direct receiver setup timed out")
            line = receiver.stdout.readline()
            if not line:
                raise RuntimeError("direct receiver failed during setup")
            info = json.loads(line)
            sender_cmd = quote_cmd(
                [
                    "python3",
                    "-c",
                    SENDER,
                    bind,
                    str(info["port"]),
                    info["fingerprint"],
                    source,
                    program,
                ]
            )
            sender = subprocess.Popen(
                ["ssh", *ssh_options(host), host, sender_cmd],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sender_err,
                text=True,
            )
            started = time.monotonic()
            output, _ = sender.communicate(token, timeout=86400)
            if sender.returncode:
                raise RuntimeError("direct sender failed")
            receive_output, _ = receiver.communicate(timeout=120)
            if receiver.returncode:
                raise RuntimeError("direct receiver failed")
            sending = json.loads(output)
            receiving = json.loads(receive_output)
            if sending["bytes"] != receiving["bytes"]:
                raise RuntimeError("direct transport byte count mismatch")
            return {
                "transport": "direct_tls",
                "bytes": receiving["bytes"],
                "seconds": time.monotonic() - started,
            }
        except BaseException:
            for process in (sender, receiver):
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
            receiver_err.seek(0)
            sender_err.seek(0)
            detail = (receiver_err.read() + sender_err.read()).decode(errors="replace").strip()
            if detail:
                raise RuntimeError(
                    "direct transfer failed: " + detail[-2000:] + " sender=" + output[-300:]
                ) from None
            raise
