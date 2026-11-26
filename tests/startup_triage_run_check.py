"""Tests for tools/startup/triage_run.py using synthetic logs and negative controls."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_STARTUP = REPO_ROOT / "tools" / "startup"
sys.path.insert(0, str(TOOLS_STARTUP))
import triage_run  # noqa: E402


SYNTHETIC_STDERR = """\
x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=unsupported_description res{dim=1 fmt=0 size=4096x1x1 mips=1 layout=0x1 flags=0x0 align=0 samples=1/0} view{dim=11 fmt=0 map=0x1688 raw=55667788,11223344,00000000,00000000,00000000,00000000 (acceleration structure views are not supported)} dest=0x00007ff712340000 dest_in_view_heap=1 address=0x00000180deadbeef address_readable=0
x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=unsupported_description res{dim=1 fmt=0 size=8192x1x1 mips=1 layout=0x1 flags=0x0 align=0 samples=1/0} view{dim=11 fmt=0 map=0x1688 raw=55667788,11223344,00000000,00000000,00000000,00000000 (acceleration structure views are not supported)} dest=0x00007ff712340080 dest_in_view_heap=1 address=0x00000180cafebabe address_readable=1
x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=destination_outside_view_heaps res{dim=3 fmt=98 size=128x128x2 mips=6 layout=0x105 flags=0x300000 align=65536 samples=1/0} view{dim=5 fmt=98 map=0x1688 mostmip=1 mips=5 first=1 count=1 plane=0 clamp=0.5} dest=0x00007ff712341001 dest_in_view_heap=0 address=0x0000018000007000 address_readable=1
x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=unreadable_resource res{unreadable} view{unreadable} dest=0x1000 dest_in_view_heap=1 address=0x0 address_readable=0
x2w-noop d3d12_xs!Device::slot44 rcx=0x0000018011112222 rdx=0x0000018033334444 r8=0x0 r9=0x0 ret=sample_app.exe+0x1234 exit=0xe257002c
x2w-trap d3d12_xs!Device::slot99 rcx=0x0000018011112222 rdx=0x0 r8=0x0 r9=0x0 ret=C:\\Users\\alice\\secret\\game.exe+0x4560 exit=0xe2570063
x2w-d3d12 Device::slot10(CreateGraphicsPipelineState) desc=[00 01] riid={1234} -> trap
x2w-d3d12-debug [Device 44] sev=2 id=999 Failed reading C:\\Users\\alice\\secret\\game.exe at 0x00007ff7deadbeef
x2w-dstorage   options=0x1 dest=0x0000018044440000 dest_size=0x1000 source=0x1 offset=0x0 size=0x100 -> zlib failed
x2w-dstorage   options=0x1 dest=0x0000018044441000 dest_size=0x1000 source=0x1 offset=0x200 size=0x100 -> zlib failed
x2w-dstorage   options=0x1 dest=0x0000018044442000 dest_size=0x1000 source=0x1 offset=0x400 size=0x100 -> done
x2w-dstorage Factory::OpenFile path="C:\\Users\\alice\\secret\\data.pak" riid={abcd} -> fail (err 2)
x2w-dstorage StatusArray::GetHResult index=0 -> 0x80004005
x2w-dstorage StatusArray::GetHResult index=1 -> 0x00000000
x2w-runtime 12345678::slot5(0x1, 0x2, 0x3 | 0x4, 0x5) -> 0xffffffff80004005 from +0x100
x2w-runtime 12345678::slot5(0x1, 0x2, 0x3 | 0x4, 0x5) -> 0x80004005 from +0x104
x2w-runtime 12345678::slot6(0x1, 0x2, 0x3 | 0x4, 0x5) -> 0x0 from +0x108
x2w-runtime persistent local storage: local object instead of 0x80070005
"""


def check_aggregation_and_redaction(failures: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        logdir = root / "logs"
        logdir.mkdir()
        (logdir / "stderr.txt").write_text(SYNTHETIC_STDERR, encoding="utf-8")

        cache_dir = root / "shader_cache"
        cache_dir.mkdir()
        (cache_dir / "0000000000000001.fail").write_text(
            "image_load_mip tfe\n", encoding="utf-8"
        )
        (cache_dir / "0000000000000002.fail").write_text(
            "image_load_mip tfe\n", encoding="utf-8"
        )
        (cache_dir / "0000000000000003.fail").write_text(
            "fxc ps: /home/alice/secret/shader.hlsl(1,1): error X3000 at 0x00007ff712345678\n",
            encoding="utf-8",
        )
        # A device slot 67 compute request with the same reason is counted in its own class.
        (cache_dir / "0000000000000004.fail").write_text("image_load_mip tfe\n", encoding="utf-8")
        (cache_dir / "req_0000000000000004.c67").write_bytes(b"\0" * 4)

        report = triage_run.triage_run(logdir, shader_cache=cache_dir, max_items=16)

        expected_snippets = [
            "Log files scanned: `1`",
            "| 2 | unsupported_description | 1 | 0 | 11 | 0 | 0x1 | 0x0 |",
            "| 1 | destination_outside_view_heaps | 3 | 98 | 5 | 98 | 0x105 | 0x300000 |",
            "| 1 | unreadable_resource | unreadable | unreadable | unreadable | unreadable | unreadable | unreadable |",
            "| 1 | `x2w-noop Device::slot44` |",
            "| 1 | `Device::slot10(CreateGraphicsPipelineState)` |",
            "| 2 | zlib failed |",
            "| 1 | 2 | 999 |",
            "| 1 | OpenFile: fail (err 2) |",
            "| 1 | `0x80004005` |",
            "| 2 | `slot5` | `0x80004005` |",
            "| 1 | `persistent_local_storage` | `0x80070005` |",
            "| 2 | `image_load_mip tfe` |",
            "| 1 | `compute: image_load_mip tfe` |",
            f"| 1 | `{triage_run.OTHER}` |",
        ]
        for snippet in expected_snippets:
            if snippet not in report:
                failures.append(f"missing expected summary line: {snippet!r}")

        # Negative controls: sensitive paths, executables, and 64-bit addresses must never leak.
        forbidden = [
            "alice",
            "game.exe",
            "sample_app.exe",
            "C:\\Users",
            "/home/alice",
            "0x00007ff712340000",
            "0x00000180deadbeef",
            "0x00007ff7deadbeef",
            "0x00007ff712345678",
            "data.pak",
        ]
        for secret in forbidden:
            if secret in report:
                failures.append(f"negative control failed: leaked {secret!r} in report")


PRIVATE_LINES = """\
x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=ACMEPrivateReason res{dim=PrivateDim fmt=acme_fmt layout=0xZZ flags=acmeflag} view{dim=x fmt=y} dest=0x1
x2w-trap acmegame_private!Device::slot31 rcx=0x1 ret=acmegame+0x10
x2w-noop ACMEOpaqueObject rcx=0x1
x2w-d3d12-debug [AcmeContext 7] sev=2 id=1234 loading AcmeTitle shader [secret](https://acme.example/leak) 0000018012345678 acme_widget_7f3a
x2w-d3d12 Device::slot44 view skipped: acme_widget_7f3a in AcmeModule.dll
x2w-d3d12 DEVICE REMOVED, reason 0x887a0005, first noticed after acme_widget_7f3a
x2w-dstorage   options=0x1 dest=0x1 dest_size=0x1 source=0x1 offset=0x0 size=0x1 -> acme_widget_7f3a exploded
x2w-dstorage entry type 3 failed: AcmeTitle bad [secret](https://acme.example/leak)
x2w-dstorage Factory::OpenFile path="x" riid={1} -> fail acme_widget_7f3a
x2w-runtime 0000018012345678::slot9(0x1) -> 0x80004005 from +0x1
x2w-runtime 0000018012345678::slot5555555555555555(0x1) -> 0x80004005
x2w-runtime 0000018012345678::slot9(0x1) -> 0x00000180DEADBEEF
"""
PRIVATE_SHADER_FAILS = [
    "AcmeModule: 0000018012345678 acme_widget_7f3a [secret](https://acme.example/leak)",
    "v_acme_widget_7f3a_private with AcmeTitle",
    "acmegame_private",
    "undecodable word at +0x10: AcmeTitle shader text",
    "image_load_mip tfe AcmeTitle",
    "image_load_mip tfe",
]
PRIVATE_NEEDLES = [
    "Acme", "acme", "ACME", "Private", "secret", "leak", "widget", "7f3a", "exploded",
    "0000018012345678", "12345678", "5555555555555555", "deadbeef", "[", "](", "http", ".dll", "Opaque",
]


def check_private_text_never_echoed(failures: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        logdir = root / "logs"
        logdir.mkdir()
        (logdir / "stderr.txt").write_text(PRIVATE_LINES, encoding="utf-8")
        cache_dir = root / "cache"
        cache_dir.mkdir()
        for index, text in enumerate(PRIVATE_SHADER_FAILS):
            (cache_dir / f"{index:016x}.fail").write_text(text + "\n", encoding="utf-8")
        report = triage_run.triage_run(logdir, shader_cache=cache_dir)
        for needle in PRIVATE_NEEDLES:
            if needle in report:
                failures.append(f"free-form text leaked into report: {needle!r}\n{report}")
        for want in [
            "| 1 | 2 | 1234 |",
            "`DEVICE REMOVED 0x887a0005`",
            "`x2w-trap Device::slot31`",
            "`x2w-noop other (text omitted)`",
            "`Device::slot44 view skipped`",
            "| 1 | OpenFile: fail (other) |",
            "| 1 | entry failed |",
            "`slot9` | `0x80004005`",
            "`undecodable word`",
            "`image_load_mip tfe`",
            "`v_* with modifier`",
            "`other (text omitted)`",
        ]:
            if want not in report:
                failures.append(f"missing grouped allowlist row: {want!r}")
        if "| 1 | other (text omitted) | ? | ? | ? | ? | ? | ? |" not in report:
            failures.append("unknown slot70 reason/values were not grouped as other/?")


def check_review_regressions(failures: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        logdir = root / "run"
        logdir.mkdir()
        fail_line = "x2w-dstorage   options=0x1 dest=0x1 dest_size=0x1000 source=0x1 offset=0x0 size=0x100 -> zlib failed\n"
        (logdir / "stderr.txt").write_text(fail_line, encoding="utf-8")
        (logdir / "trap.log").write_text(fail_line, encoding="utf-8")
        report = triage_run.triage_run(logdir)
        if "| 1 | zlib failed |" not in report:
            failures.append("mirrored stderr.txt/trap.log line counted other than once:\n" + report)
        # A genuine repeat (second occurrence only in trap.log) still counts.
        (logdir / "trap.log").write_text(fail_line * 2, encoding="utf-8")
        report = triage_run.triage_run(logdir)
        if "| 2 | zlib failed |" not in report:
            failures.append("repeated trap.log occurrence beyond stderr.txt was dropped:\n" + report)
    # Successful forwarded slots are not "unhandled"; failed ones and untraced calls still are.
    lines = [
        "x2w-d3d12 Device::slot13(0x1, 0x2, 0x3, 0x4, 0x5) -> 0x0\n",
        "x2w-d3d12 Device::slot41(0x1, 0x2, 0x3, 0x4, 0x5) -> 0x0\n",
        "x2w-d3d12 Device::slot14(0x1, 0x2, 0x3, 0x4, 0x5) -> 0x80070057\n",
        "x2w-d3d12 Device::slot15 not implemented\n",
    ]
    unhandled = triage_run.summarize_lines(lines)["d3d12_unhandled"]
    if "Device::slot13" in unhandled or "Device::slot41" in unhandled:
        failures.append(f"successful forwarded slots counted as unhandled: {dict(unhandled)}")
    if "Device::slot14" not in unhandled or "Device::slot15" not in unhandled:
        failures.append(f"failed/untraced slots missing from unhandled: {dict(unhandled)}")


def check_every_slot70_reason_is_reported(failures: list[str]) -> None:
    # Each reason the producer can log keeps its own row; none is folded into "other".
    cases = [
        ("unsupported memory SRV arguments", "null_destination"),
        ("unsupported memory SRV arguments", "unreadable_resource"),
        ("unsupported memory SRV arguments", "unreadable_view"),
        ("unsupported memory SRV arguments", "null_address"),
        ("unsupported memory SRV arguments", "unsupported_description"),
        ("unsupported memory SRV arguments", "destination_outside_view_heaps"),
        ("GPU-written resource at the supplied address", "gpu_written_resource"),
        ("unavailable acceleration structure at the supplied address", "as_unaligned"),
        ("unavailable acceleration structure at the supplied address", "as_not_placed"),
        ("unavailable acceleration structure at the supplied address", "as_wrong_state"),
        ("unavailable buffer at the supplied address", "buffer_unavailable"),
        ("unavailable texture at the supplied base and layout", "texture_unavailable"),
        ("unavailable texture at the supplied base and layout", "texture_no_resource"),
        ("unavailable texture at the supplied base and layout", "texture_unreadable"),
        ("unavailable texture at the supplied base and layout", "texture_over_64mib"),
        ("unavailable texture at the supplied base and layout", "partial_tile_xg_rows"),
        ("unavailable texture at the supplied base and layout", "partial_tile_beyond_xg_size"),
        ("unavailable texture at the supplied base and layout", "unsupported_desktop_format"),
        ("unavailable texture at the supplied base and layout", "console_layout_xg_rows"),
        ("unavailable texture at the supplied base and layout", "console_layout_without_footprints"),
        ("unavailable texture at the supplied base and layout", "console_layout_beyond_xg_size"),
        ("unavailable texture at the supplied base and layout", "overlapping_resource"),
        ("unsupported memory SRV arguments", "volume_texture_disabled"),
        ("unsupported memory SRV arguments", "volume_unsupported_swizzle"),
        ("unsupported memory SRV arguments", "volume_unsupported_format"),
        ("unsupported memory SRV arguments", "volume_requires_one_mip"),
        ("unsupported memory SRV arguments", "volume_invalid_dimensions"),
        ("unsupported memory SRV arguments", "volume_invalid_samples"),
        ("unsupported memory SRV arguments", "volume_unsupported_flags"),
        ("unsupported memory SRV arguments", "volume_unsupported_alignment"),
        ("unsupported memory SRV arguments", "volume_incompatible_view"),
    ]
    reasons = [reason for _, reason in cases]
    lines = [
        f"x2w-d3d12 Device::slot70 {what}: reason={reason} "
        "res{dim=3 fmt=98 size=128x128x2 mips=6 layout=0x105 flags=0x300000 align=65536 samples=1/0} "
        "view{dim=5 fmt=98 map=0x1688} dest=0x1 dest_in_view_heap=1 address=0x2 address_readable=1\n"
        for what, reason in cases
    ]
    refusals = triage_run.summarize_lines(lines)["slot70_refusals"]
    reported = sorted(key[0] for key in refusals)
    if reported != sorted(reasons):
        failures.append(f"slot70 reasons reported as {reported}, expected {sorted(reasons)}")
    # The acceleration-structure names come from the header, so a new one cannot be forgotten here.
    as_header = (Path(__file__).resolve().parent.parent / "src" / "startup_stubs" / "memory_srv_as.h").read_text(encoding="utf-8")
    body = as_header.split("as_srv_refusal_name", 1)[1].split("}", 1)[0]
    as_names = re.findall(r'return "(as_\w+)";', body)
    if len(as_names) < 3:
        failures.append(f"as_srv_refusal_name names not found in memory_srv_as.h: {as_names}")
    for name in as_names:
        if name not in triage_run.SLOT70_REASONS:
            failures.append(f"SLOT70_REASONS lacks acceleration-structure refusal {name}")


def check_empty_and_missing_dirs(failures: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        empty_dir = root / "empty"
        empty_dir.mkdir()
        missing_dir = root / "does_not_exist"

        for target in (empty_dir, missing_dir):
            report = triage_run.triage_run(target, shader_cache=missing_dir)
            if "Log files scanned: `0`" not in report:
                failures.append(f"expected 0 scanned files for {target}")
            if "_None observed._" not in report:
                failures.append(f"expected '_None observed._' for {target}")


def check_slot133_outcomes(failures: list[str]) -> None:
    # Device::slot133 logs every handled placement with a trailing "aliased|not aliased gpuva=..." field.
    prefix = "x2w-d3d12 Device::slot133(CreatePlacedResourceX?) "
    lines = [
        prefix + "address=0x1000 base=0x1000 offset=0x0 width=0x100 flags=0x0 state=0x0 clear=0x0 riid=ID3D12Resource -> 0x00000000 aliased gpuva=0x2000\n",
        prefix + "address=0x3000 base=0x3000 offset=0x0 width=0x100 flags=0x0 state=0x0 clear=0x0 riid=ID3D12Resource -> 0x00000000 not aliased gpuva=0x4000\n",
        prefix + "address=0x5000 base=0x5000 offset=0x0 width=0x100 flags=0x0 state=0x0 clear=0x0 riid=ID3D12Resource -> 0x80070057 not aliased gpuva=0x0\n",
        prefix + "texture address=0x6000 dim=3 64x64x1 mips=1 fmt=28 samples=1 flags=0x0 layout=0x0 state=0x0 clear=0x0 -> 0x00000000 not aliased []\n",
        prefix + "address=0x7000 not committed -> E_INVALIDARG\n",
        "x2w-d3d12 Device::slot15 not implemented\n",
    ]
    summary = triage_run.summarize_lines(lines)
    outcomes = dict(summary["slot133_outcomes"])
    if outcomes != {"aliased": 1, "not_aliased": 2, "failed": 2}:
        failures.append(f"slot133 outcomes wrong: {outcomes}")
    unhandled = dict(summary["d3d12_unhandled"])
    if unhandled != {"Device::slot133": 2, "Device::slot15": 1}:
        failures.append(f"slot133 successes counted as unhandled, or failures dropped: {unhandled}")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "stderr.txt").write_text("".join(lines), encoding="utf-8")
        report = triage_run.triage_run(Path(tmp))
    if "| 2 | not_aliased |" not in report or "| 1 | aliased |" not in report or "| 2 | failed |" not in report:
        failures.append("slot133 outcome table missing from the report:\n" + report)
    if "gpuva" in report or "0x4000" in report:
        failures.append("slot133 addresses leaked into the report:\n" + report)
    # Control: with the old classification (no slot133 rule) the same log counts the successes as unhandled.
    saved = (triage_run.RE_D3D12_SLOT133, triage_run.RE_D3D12_SLOT133_UNCOMMITTED)
    never = re.compile(r"(?!)")
    triage_run.RE_D3D12_SLOT133 = triage_run.RE_D3D12_SLOT133_UNCOMMITTED = never
    try:
        old = triage_run.summarize_lines(lines)["d3d12_unhandled"]
    finally:
        triage_run.RE_D3D12_SLOT133, triage_run.RE_D3D12_SLOT133_UNCOMMITTED = saved
    if old.get("Device::slot133") != 5:
        failures.append(f"control did not reproduce the old unhandled noise: {dict(old)}")


def check_cli_invocation(failures: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log_file = Path(tmp) / "single.log"
        log_file.write_text(
            "x2w-dstorage   options=0x1 dest=0x0000018000001000 dest_size=0x10 "
            "source=0x1 offset=0x0 size=0x10 -> read failed\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, "-B", str(TOOLS_STARTUP / "triage_run.py"), str(log_file), "--max-items", "4"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            failures.append(f"CLI exited with {proc.returncode}: {proc.stderr}")
        elif "| 1 | read failed |" not in proc.stdout:
            failures.append(f"CLI output missing expected row: {proc.stdout!r}")


HEARTBEAT_FAILURES = (
    "x2w-dstorage heartbeat {second} s: failed requests by cause read-at-or-past-eof={a} read-crossing-eof=0 "
    "read-other={b} destination-capacity=3 destination-not-writable=0 zlib=11 bcpack=0 other=1\n"
)
HEARTBEAT_COMMIT = (
    "x2w-dstorage heartbeat {second} s: requests done=10, commit peak={peak} MiB current={current} MiB, "
    "live physical-page sections=4\n"
)
ENDPOINT_EVENTS = """\
  0.120 module load C:\\Users\\alice\\secret\\game.exe base=0x00007ff700000000
  1.500 exception 0x80000003 breakpoint at ntdll.dll+0x1234 first-chance params=[]
  2.000 exception 0x406d1388 thread name at game.exe+0x10 first-chance params=['0x1']
  250.250 exception 0xE06D7363 C++ exception at C:\\Users\\alice\\secret\\game.exe+0x55 first-chance params=['0x19930520', '0x00007ff7deadbeef']
  299.900 exception 0xC0000005 access violation at KERNELBASE.dll+0x9000 second-chance params=['0x0', '0x00007ff7cafebabe']
"""
ENDPOINT_FORBIDDEN = [
    "alice", "secret", "game.exe", "KERNELBASE", "ntdll", "deadbeef", "cafebabe", "0x19930520", "00007ff", "params",
]


def _endpoint_report(logs: dict[str, str]) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        logdir = Path(tmp)
        for name, text in logs.items():
            (logdir / name).write_text(text, encoding="utf-8")
        return triage_run.triage_run(logdir)


def _check_endpoint_complete(failures: list[str]) -> None:
    report = _endpoint_report(
        {
            "stderr.txt": HEARTBEAT_COMMIT.format(second=20, peak=900, current=800)
            + HEARTBEAT_FAILURES.format(second=20, a=1, b=2)
            + HEARTBEAT_COMMIT.format(second=290, peak=16208, current=15000)
            + HEARTBEAT_COMMIT.format(second=299, peak=14000, current=13000)  # a lower peak after a higher one
            + HEARTBEAT_FAILURES.format(second=299, a=7, b=9)
            + HEARTBEAT_FAILURES.format(second=50, a=99, b=99),
            "events.txt": ENDPOINT_EVENTS,
        }
    )
    for want in [
        "### Run End-Point",
        "| Peak commit (maximum of the `commit peak=` values) | `16208` MiB |",
        "| Last heartbeat | `299` s |",
        "| Confirmed debugger stop | not reported |",
        "| First observed exception | `0xe06d7363` (first-chance) |",
        "| Last observed exception | `0xc0000005` (second-chance) |",
        "| Observed exceptions | `2` |",
        "at heartbeat second `299`",
        "| read-at-or-past-eof | 7 |",
        "| read-other | 9 |",
        "| destination-capacity | 3 |",
        "| zlib | 11 |",
    ]:
        if want not in report:
            failures.append(f"run end-point: missing {want!r}\n{report}")
    for secret in ENDPOINT_FORBIDDEN:
        if secret in report:
            failures.append(f"run end-point leaked {secret!r}")
    for stepped_over in ("0x80000003", "0x406d1388"):
        if stepped_over in report.lower():
            failures.append(f"run end-point reported a stepped-over exception {stepped_over}")


def _check_endpoint_without_commit(failures: list[str]) -> None:
    report = _endpoint_report({"stderr.txt": HEARTBEAT_FAILURES.format(second=40, a=0, b=0)})
    for want in [
        "| Peak commit (maximum of the `commit peak=` values) | not reported |",
        "| Last heartbeat | `40` s |",
        "| Confirmed debugger stop | not reported |",
        "| Observed exceptions | `0` |",
        "at heartbeat second `40`",
    ]:
        if want not in report:
            failures.append(f"run end-point without commit line: missing {want!r}")
    report = _endpoint_report({"stderr.txt": "x2w-dstorage heartbeat 12 s: requests done=1, commit unknown, live physical-page sections=unknown\n"})
    if "| Peak commit (maximum of the `commit peak=` values) | not reported |" not in report or "| Last heartbeat | `12` s |" not in report:
        failures.append("run end-point: a 'commit unknown' heartbeat was not handled")
    if "_No failed-requests-by-cause heartbeat line._" not in report:
        failures.append("run end-point: missing the note for a run with no cause line")
    report = _endpoint_report({})
    if "| Last heartbeat | not reported |" not in report:
        failures.append("run end-point: an empty run must report no heartbeat")


def _check_endpoint_second_stop(failures: list[str]) -> None:
    events = (
        "  10.000 exception 0xC0000005 access violation at a.dll+0x1 first-chance params=[]\n"
        "  11.000 exception 0xC0000005 access violation at a.dll+0x1 second-chance params=[]\n"
        "debugger stop 0xc0000005 second-chance\n"
    )
    report = _endpoint_report({"events.txt": events})
    for want in [
        "| First observed exception | `0xc0000005` (first-chance) |",
        "| Last observed exception | `0xc0000005` (second-chance) |",
        "| Confirmed debugger stop | `0xc0000005` (second-chance) |",
        "| Observed exceptions | `2` |",
    ]:
        if want not in report:
            failures.append(f"run end-point with a second stop: missing {want!r}")


def _check_endpoint_breakpoint(failures: list[str]) -> None:
    events = (
        "  1.000 exception 0x80000003 breakpoint at ntdll.dll+0x13 first-chance params=[]\n"
        "  2.000 exception 0x80000003 breakpoint at app.exe+0x20 first-chance params=[]\n"
    )
    modern = _endpoint_report({"events.txt": events + "debugger stop 0x80000003 first-chance\n"})
    legacy = _endpoint_report({
        "events.txt": events,
        "run.out": "stopped at breakpoint (0x80000003) at app.exe+0x20 after 2.00 s (first chance) params=[]\n",
    })
    unfinished = _endpoint_report({"events.txt": events})
    for report in (modern, legacy):
        if "| Confirmed debugger stop | `0x80000003` (first-chance) |" not in report:
            failures.append("run end-point: stopping breakpoint was lost")
        if "| Observed exceptions | `0` |" not in report or "app.exe" in report:
            failures.append("run end-point: loader or assertion breakpoint details were counted or copied")
    if "| Confirmed debugger stop | not reported |" not in unfinished:
        failures.append("run end-point: first-chance event alone was classified as a stop")


def _check_endpoint_private_lines(failures: list[str]) -> None:
    hostile = (
        "  5.000 exception 0xE06D7363deadbeef C++ exception at X first-chance params=[]\n"  # not a 32-bit code
        "  6.000 exception 0xZZ00ZZ00 junk at Y second-chance\n"
        "  7.000 exception 0xC0000005 at C:\\Users\\alice\\x.dll+0x1 [secret](https://acme.example/leak) second-chance\n"
        "x2w-dstorage heartbeat 30 s: failed requests by cause read-other=5 acme_widget=99 zlib=0x7ff7deadbeef other=2\n"
        "x2w-dstorage heartbeat 31 s: requests done=1, commit peak=0x00007ff7deadbeef MiB current=1 MiB, live physical-page sections=1\n"
    )
    report = _endpoint_report({"events.txt": hostile})
    for needle in ["alice", "secret", "acme", "widget", "deadbeef", "leak", "http", "99 |"]:
        if needle in report:
            failures.append(f"run end-point copied hostile text {needle!r}\n{report}")
    if "| Observed exceptions | `1` |" not in report or "`0xc0000005` (second-chance)" not in report:
        failures.append("run end-point: only the well-formed 32-bit code should be reported")
    if "| Peak commit (maximum of the `commit peak=` values) | not reported |" not in report:
        failures.append("run end-point: a non-decimal peak must not be reported")


def _endpoint_failures() -> list[str]:
    failures: list[str] = []
    for check in (
        _check_endpoint_complete,
        _check_endpoint_without_commit,
        _check_endpoint_second_stop,
        _check_endpoint_breakpoint,
        _check_endpoint_private_lines,
    ):
        check(failures)
    return failures


def check_run_endpoint(failures: list[str]) -> None:
    failures.extend(_endpoint_failures())
    # Mutations: each wrong behaviour must make the end-point checks fail.
    mutations = {
        "peak taken from the last line instead of the maximum": ("_keep_peak", lambda old, new: new),
        "an earlier cause line replaces a later one": ("_is_later_sample", lambda old, new: old is None),
        "an out-of-order cause line replaces a later one": ("_is_later_sample", lambda old, new: True),
        "the exception code is not limited to 32 bits": (
            "RE_EXCEPTION",
            re.compile(r"^\s*\d+\.\d+\s+exception\s+(0x[0-9a-fA-F]+)\b.*?\b(first|second)-chance\b"),
        ),
        "stepped-over exceptions are counted": ("FILTERED_FIRST_CHANCE", set()),
    }
    for label, (name, replacement) in mutations.items():
        original = getattr(triage_run, name)
        setattr(triage_run, name, replacement)
        try:
            caught = bool(_endpoint_failures())
        finally:
            setattr(triage_run, name, original)
        if not caught:
            failures.append(f"mutation not caught by the run end-point checks: {label}")

def check_every_shader_reason_entry_is_reachable(failures: list[str]) -> None:
    # Each table entry is a distinct report token. A pattern that can never match, or a label whose
    # text changed, folds a known cause into "other" without failing anything else. The samples are
    # matched back to the table so a new entry without a sample fails here.
    samples = (
        ("image_load_mip tfe", "image_load_mip tfe"),
        ("image_load_mip a16", "image_load_mip a16"),
        ("buffer_load_format_xyzw requires descriptor FORMAT and DST_SEL", "buffer_* instruction"),
        ("SDWA UNUSED_SEXT", "SDWA UNUSED_SEXT"),
        ("SDWA dst_unused:UNUSED_PRESERVE needs a tracked destination", "SDWA dst_unused"),
        ("undecodable word at +0x10", "undecodable word"),
        ("numeric use of s0 holding a descriptor", "numeric use of scalar register"),
        ("operand v[0:3] is not a single register", "unsupported operand"),
        ("modifier r128 is not lowered", "unsupported modifier"),
        ("descriptor operand s[0:7]", "descriptor operand"),
        ("sampler operand s[8:11]", "sampler operand"),
        ("mask from descriptor data", "mask from descriptor data"),
        ("s_load of 4 dwords from an unaligned address", "s_load form"),
        ("v_mov_b32 SDWA sel_enc", "v_* SDWA form"),
        ("v_mov_b32 with clamp", "v_* with modifier"),
        ("v_mov_b32", "v_* instruction"),
    )
    covered: set[int] = set()
    for line, expected in samples:
        got = triage_run.classify_shader_reason(line)
        if got != expected:
            failures.append(f"classify_shader_reason({line!r}) -> {got!r}, expected {expected!r}")
        for index, (pattern, _) in enumerate(triage_run.SHADER_REASONS):
            if pattern.match(line):
                covered.add(index)
                break
    missing = [index for index in range(len(triage_run.SHADER_REASONS)) if index not in covered]
    for index in missing:
        failures.append(
            f"SHADER_REASONS[{index}] ({triage_run.SHADER_REASONS[index][0].pattern!r}) has no sample line"
        )
    if triage_run.classify_shader_reason("a reason nobody modelled") != triage_run.OTHER:
        failures.append("an unmodelled reason did not fall back to the other token")

def check_indirect_outcomes(failures: list[str]) -> None:
    initial = (
        "x2w-render indirect: forwarded: draw 2; refused graphics-pipeline: dispatchmesh 5; "
        "compute-pipeline: dispatch 3; placeholder-signature: drawindexed 1; other: other 1"
    )
    later = initial.replace("dispatchmesh 5", "dispatchmesh 12")
    older = initial.replace("dispatchmesh 5", "dispatchmesh 4")
    invalid = (
        later.replace("other: other 1", r"other: C:\Users\alice\secret.exe"),
        later.replace("draw 2", "draw 2 draw 99"),
        later.replace("dispatchmesh 12", "dispatchmesh 99999999999999999999"),
        later.replace("dispatchmesh 12", "dispatchmesh -99"),
        later.replace("; other: other 1", ""),
        later.replace("forwarded: draw 2", "forwarded: draw 2 dispatchmesh 0x00007ff712345678"),
    )
    zero = (
        "x2w-render indirect: forwarded: 0; refused graphics-pipeline: 0; "
        "compute-pipeline: 0; placeholder-signature: 0; other: 0"
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "stderr.txt").write_text("\n".join((initial, later, later, *invalid)), encoding="utf-8")
        (root / "trap.log").write_text(initial + "\n" + older + "\n", encoding="utf-8")
        report = triage_run.triage_run(root)
        expected = (
            "### `ExecuteIndirect` Outcomes",
            "| forwarded | 0 | 0 | 2 | 0 | 0 |",
            "| refused graphics-pipeline | 0 | 0 | 0 | 0 | 12 |",
            "| compute-pipeline | 0 | 3 | 0 | 0 | 0 |",
            "| placeholder-signature | 0 | 0 | 0 | 1 | 0 |",
            "| other | 1 | 0 | 0 | 0 | 0 |",
        )
        for snippet in expected:
            if snippet not in report:
                failures.append(f"missing indirect summary line: {snippet!r}")
        for secret in ("alice", "secret.exe", "0x00007ff712345678"):
            if secret in report:
                failures.append(f"indirect summary copied private text: {secret!r}")
        log = root / "stderr.txt"
        log.write_text(zero + "\n", encoding="utf-8")
        if "| forwarded | 0 | 0 | 0 | 0 | 0 |" not in triage_run.triage_run(log):
            failures.append("a complete zero snapshot was not reported")
        log.write_text("\n".join(invalid), encoding="utf-8")
        if "### `ExecuteIndirect` Outcomes" in triage_run.triage_run(log):
            failures.append("a malformed snapshot produced an indirect summary")


def check_gpu_memory_budget(failures: list[str]) -> None:
    def sample(local, texture, commit=""):
        return (
            f'x2w-budget local={local} nonlocal=2/8 placed=10000/2+20000/3 texture={texture} '
            f'buffer=1/1+0/0 upload=0/0+0/0 readback=0/0+0/0 heap=0/0+0/0 unmeasured=0{commit}'
        )

    # The line the shim writes now carries the system-memory category and the fallback totals: it still parses.
    current = sample("2500/3000", "100/5+40/2").replace(" unmeasured=0", " sysmem=4096/1+0/0 unmeasured=0 sysmem-fallback=1 bytes=4096")
    if not triage_run.RE_GPU_BUDGET.fullmatch(current):
        failures.append("the budget line with the sysmem category and fallback totals was not parsed")
    # The system commit and pool counters report_budget() appends (known, and unavailable).
    with_commit = " commit=14335/16100 avail=1765 sections=113609 slabs=3 slots=113000/609 job_peak=14250 job_limit=14336"
    commit_n_a = " commit=n/a avail=n/a sections=n/a slabs=n/a slots=n/a job_peak=n/a job_limit=none"

    high_usage = sample("2500/3000", "100/5+40/2")
    high_ratio = sample("1800/1000", "150/7+50/3")
    high_retained = sample("n/a", "250/8+900/40").replace("nonlocal=2/8", "nonlocal=n/a")
    malformed = (
        sample("4000/1", "500/7+950/4").replace(" unmeasured=0", ""),
        sample("9000/1", "900/1+999/1").replace(" buffer=1/1+0/0", r" buffer=C:\Users\alice"),
        sample("9000/1", "900/1+999/1") + " path=secret.exe",
        sample("9000/1", "900/1+999/1", " commit=14335/x avail=1765 sections=1 slabs=1 slots=1/1"),
        sample("9000/1", "900/1+999/1", with_commit + " path=secret.exe"),
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        log = root / "stderr.txt"
        samples = (high_usage, high_ratio, sample("2100/1800", "170/8+80/4", with_commit), high_retained,
                   sample("5000/0", "300/9+10/3", commit_n_a), sample("1000/1000", "90/6+20/1"), *malformed)
        log.write_text("\n".join(samples), encoding="utf-8")
        (root / "trap.log").write_text(high_ratio + "\n", encoding="utf-8")
        report = triage_run.triage_run(root)
        for snippet in (
            "### GPU Memory Budget",
            "Complete samples: `6`",
            "Samples above local budget: `2` of `4`",
            "| Highest local usage-to-budget ratio | 1800 | 1000 | 150 | 50 | 3 |",
            "| Most retained texture bytes | n/a | n/a | 250 | 900 | 40 |",
        ):
            if snippet not in report:
                failures.append(f"missing memory budget line: {snippet!r}")
        for private in ("alice", "secret.exe"):
            if private in report:
                failures.append(f"memory budget copied private text: {private!r}")
        log.write_text("\n".join(malformed), encoding="utf-8")
        if "### GPU Memory Budget" in triage_run.triage_run(log):
            failures.append("incomplete or malformed budget rows produced a section")
        log.write_text(high_retained, encoding="utf-8")
        report = triage_run.triage_run(log)
        if "Highest local usage-to-budget ratio" in report or "Most retained texture bytes" not in report:
            failures.append("an unavailable DXGI budget hid the retained-texture sample or produced a ratio")


def main() -> int:
    failures: list[str] = []
    check_aggregation_and_redaction(failures)
    check_private_text_never_echoed(failures)
    check_review_regressions(failures)
    check_every_slot70_reason_is_reported(failures)
    check_empty_and_missing_dirs(failures)
    check_cli_invocation(failures)
    check_slot133_outcomes(failures)
    check_indirect_outcomes(failures)
    check_gpu_memory_budget(failures)
    check_run_endpoint(failures)
    check_every_shader_reason_entry_is_reachable(failures)
    for item in failures:
        print("FAIL", item)
    print(f"startup triage_run checks: {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
