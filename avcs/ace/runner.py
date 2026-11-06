"""Autonomous Visual GUI Autopilot Runner (Ace-Control-Medium).

Executes the continuous observe() -> plan() -> execute() loop with:
1. Native Win32 5th-order minimum-jerk hardware actuation.
2. Direct Action Token decoding with zero CoT latency overhead.
3. Fail-safe corner breakout & emergency abort mechanisms.
4. Non-blocking natural speech feedback on lifecycle events.
"""
from __future__ import annotations

import argparse
import ctypes
import logging
import sys
import time
from typing import Optional, Tuple

from avcs.ace.actions import Action, ActionKind
from avcs.ace.agent import AceAgent
from avcs.ace.client import CustomLLMClient
from avcs.ace.computer import WindowsComputer
from avcs.ace.session import AceSession
from avcs.ace.streamer import ScreenStreamer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [ACE] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("avcs.ace.runner")


class SafetyBreakoutError(Exception):
    """Raised when emergency fail-safe is triggered by user intervention."""
    pass


_tts_speaker = None


def speak_feedback(text: str, enabled: bool = True) -> None:
    """Delivers natural speech feedback via AVCS TTS or graceful console fallback."""
    if not enabled or not text:
        return
    logger.info(f"VOICE: {text}")
    try:
        from avcs.voice.tts import Speaker
        global _tts_speaker
        if _tts_speaker is None:
            _tts_speaker = Speaker()
        _tts_speaker.speak_async(text)
    except Exception as e:
        logger.debug(f"Speech fallback: {e}")


class AceRunner:
    """Continuous Autonomous Execution Loop for Desktop Autopilot."""

    def __init__(
        self,
        agent: Optional[AceAgent] = None,
        computer: Optional[WindowsComputer] = None,
        enable_voice: bool = True,
    ) -> None:
        self.computer = computer or WindowsComputer()
        self.agent = agent or AceAgent()
        self.enable_voice = enable_voice
        self.user32 = getattr(ctypes.windll, "user32", None) if sys.platform == "win32" else None
        self._last_actuation_coord: Optional[Tuple[int, int]] = None

    def check_safety_breakout(self, target_coord: Optional[Tuple[int, int]] = None) -> None:
        """Emergency failsafe check: corner mouse breakout in any of the 4 screen corners or ESC key.

        Exempts intentional corner actions targeting bottom-left (Start button) or
        top-right (window Close button).
        """
        if self.user32 is None:
            return

        # 1. 4-Corner mouse breakout (margin 15px)
        try:
            cx, cy = self.computer.get_cursor_pos()
            nw = self.computer.native_width
            nh = self.computer.native_height
            margin = 15

            top_left = 0 <= cx <= margin and 0 <= cy <= margin
            top_right = (nw - margin) <= cx <= nw and 0 <= cy <= margin
            bottom_left = 0 <= cx <= margin and (nh - margin) <= cy <= nh
            bottom_right = (nw - margin) <= cx <= nw and (nh - margin) <= cy <= nh

            # Check if this is an intentional corner action (e.g. Start button or Close button)
            exempt_bottom_left = False
            exempt_top_right = False

            check_coords = []
            if target_coord is not None:
                check_coords.append(target_coord)
            if self._last_actuation_coord is not None:
                check_coords.append(self._last_actuation_coord)

            for tx, ty in check_coords:
                if 0 <= tx <= margin * 4 and (nh - margin * 4) <= ty <= nh:
                    exempt_bottom_left = True
                if (nw - margin * 4) <= tx <= nw and 0 <= ty <= margin * 4:
                    exempt_top_right = True

            if top_left:
                raise SafetyBreakoutError(f"Safety breakout: Mouse pushed to top-left corner ({cx}, {cy}). Aborting immediately.")
            elif top_right and not exempt_top_right:
                raise SafetyBreakoutError(f"Safety breakout: Mouse pushed to top-right corner ({cx}, {cy}). Aborting immediately.")
            elif bottom_left and not exempt_bottom_left:
                raise SafetyBreakoutError(f"Safety breakout: Mouse pushed to bottom-left corner ({cx}, {cy}). Aborting immediately.")
            elif bottom_right:
                raise SafetyBreakoutError(f"Safety breakout: Mouse pushed to bottom-right corner ({cx}, {cy}). Aborting immediately.")
        except Exception as e:
            if isinstance(e, SafetyBreakoutError):
                raise

        # 2. ESC key emergency abort (VK_ESCAPE = 0x1B)
        try:
            if self.user32.GetAsyncKeyState(0x1B) & 0x8000:
                raise SafetyBreakoutError("Safety breakout: ESC key pressed. Aborting autopilot.")
        except Exception as e:
            if isinstance(e, SafetyBreakoutError):
                raise

    def run(
        self,
        goal: str,
        max_steps: int = 30,
        image_format: str = "jpeg",
        dry_run: bool = False,
    ) -> AceSession:
        """Executes the autonomous loop until completion or maximum steps."""
        session = AceSession()
        session.start(goal=goal, max_steps=max_steps, dry_run=dry_run)

        mode_str = "[DRY-RUN]" if dry_run else "[LIVE-ACTUATION]"
        logger.info(f"Starting Ace-Control-Medium Autopilot {mode_str}")
        logger.info(f"Goal: '{goal}' | Max Steps: {max_steps} | Model: {self.agent.client.model}")

        if dry_run:
            logger.warning(
                "[DRY-RUN MODE ACTIVE] Physical mouse and keyboard hardware events are intentionally BYPASSED. "
                "The desktop display will NOT change automatically unless you perform actions manually. "
                "To allow the autopilot to actuate on your live Windows desktop, run without --dry-run."
            )
            speak_feedback(
                "Dry run mode active. Mouse and keyboard actions are simulated and will not change your screen.",
                enabled=self.enable_voice,
            )
        else:
            speak_feedback(f"Starting desktop autopilot for: {goal}", enabled=self.enable_voice)

        streamer = ScreenStreamer(computer=self.computer, target_fps=30.0, image_format=image_format)
        step_idx = 1
        try:
            streamer.start()
            while not session.is_complete and step_idx <= max_steps:
                logger.info(f"--- Step {step_idx}/{max_steps} ---")

                # 1. Safety Breakout Check
                self.check_safety_breakout()

                # 2. Visual Desktop Observation (Instant stream frame <0.1ms)
                obs = streamer.get_latest_observation() or self.computer.observe(format=image_format)
                logger.info(f"Observed desktop: {obs.width}x{obs.height} ({obs.format}) in {obs.capture_time_ms:.1f}ms")

                # 3. Direct Action Token Inference (~533ms baseline)
                history_summary = session.get_history_summary()
                try:
                    action, plan_latency_ms = self.agent.step(
                        observation=obs,
                        goal=goal,
                        history=history_summary,
                        step_index=step_idx,
                    )
                except Exception as e:
                    logger.error(f"Planning error at step {step_idx}: {e}")
                    action = Action(kind=ActionKind.WAIT, duration=1.0, reason=f"Planning error: {e}")
                    plan_latency_ms = 0.0

                # Repetition & Stuck Detection / Jittering
                repetition_count = session.get_repetition_count(new_action=action)
                recent_matches = session.count_recent_matches(action, window=5)
                effective_reps = max(repetition_count, recent_matches)
                if effective_reps >= 2:
                    if dry_run:
                        logger.warning(
                            f"[DRY-RUN] Repetition detected ({effective_reps + 1}x for {action.kind.value} {action.coordinate or ''}). "
                            "Screen state remains static because hardware actuation is bypassed in dry-run mode. "
                            "To allow the autopilot to actuate on your live Windows desktop, run without --dry-run."
                        )
                    if action.kind in (
                        ActionKind.LEFT_CLICK,
                        ActionKind.RIGHT_CLICK,
                        ActionKind.DOUBLE_CLICK,
                        ActionKind.TRIPLE_CLICK,
                    ) and action.coordinate:
                        orig_x, orig_y = action.coordinate
                        jitter_x = 10 if orig_x < 1270 else -10
                        jitter_y = -10 if orig_y >= 790 else 10
                        new_coord = (orig_x + jitter_x, orig_y + jitter_y)
                        logger.warning(
                            f"Loop detected: Click action repeated {effective_reps + 1} times. "
                            f"Jittering position from {action.coordinate} to {new_coord}."
                        )
                        action = Action(
                            kind=action.kind,
                            coordinate=new_coord,
                            reason=f"Loop breakout: Jittering click position from {action.coordinate} to {new_coord}",
                        )

                # Execution Safety Firewall: Reject typing into terminal if coordinate is omitted
                if action.kind == ActionKind.TYPE and action.coordinate is None and self.user32:
                    fg_hwnd = self.user32.GetForegroundWindow()
                    if fg_hwnd:
                        buf = ctypes.create_unicode_buffer(256)
                        self.user32.GetClassNameW(fg_hwnd, buf, 256)
                        cls_name = buf.value
                        if any(term in cls_name for term in ("CASCADIA", "ConsoleWindowClass", "WindowsTerminal")):
                            logger.warning(
                                f"Execution Safety Firewall: Active console '{cls_name}' detected. "
                                "Rejecting ungrounded typing into terminal to prevent corruption. Click target UI element first."
                            )
                            action = Action(
                                kind=ActionKind.WAIT,
                                duration=0.5,
                                reason=f"Execution Safety Firewall: Active console '{cls_name}' detected. Rejecting ungrounded typing into terminal to prevent corruption. Click target UI element first.",
                            )

                logger.info(f"Planned Action: {action.kind.value} {action.coordinate or ''} | Reason: {action.reason} | Latency: {plan_latency_ms:.1f}ms")

                # 4. Record step in session (includes honest reflection feedback if firewall triggered)
                session.record_step(action, observation=obs, plan_latency_ms=plan_latency_ms)

                # 5. Check Termination Condition
                if action.kind == ActionKind.STOP:
                    logger.info(f"Goal completed: {action.reason}")
                    speak_feedback(f"Task completed: {action.reason}", enabled=self.enable_voice)
                    break

                # 6. Physical Win32 Hardware Execution (Minimum-Jerk)
                # Map target coordinate for intentional corner exemption & tracking
                native_target = None
                if action.coordinate:
                    native_target = self.computer._canonical_to_native(*action.coordinate)
                elif action.start_coordinate:
                    native_target = self.computer._canonical_to_native(*action.start_coordinate)

                self.check_safety_breakout(target_coord=native_target)

                if dry_run:
                    logger.info(f"[DRY-RUN] Simulated execution for {action.kind.value} {action.coordinate or ''} (hardware actuation bypassed)")
                    self.computer.execute(action, dry_run=True)
                else:
                    self.computer.execute(action, dry_run=False)

                if native_target is not None:
                    self._last_actuation_coord = native_target

                # Visual closed-loop delta verification: probe screen change after click
                if not dry_run and action.kind in (
                    ActionKind.LEFT_CLICK,
                    ActionKind.RIGHT_CLICK,
                    ActionKind.DOUBLE_CLICK,
                    ActionKind.TRIPLE_CLICK,
                ):
                    time.sleep(0.04)
                    obs_after = self.computer.observe(format=image_format)
                    has_delta = (obs_after.image_bytes != obs.image_bytes)
                    if not has_delta:
                        logger.warning(
                            f"Visual closed-loop verification: No screen change detected after {action.kind.value} at {action.coordinate}."
                        )
                    else:
                        logger.info(
                            f"Visual closed-loop verification: Visual delta confirmed after {action.kind.value} at {action.coordinate}."
                        )

                # 7. Post-action settling delay (allow Start flyout animation to complete if Win key)
                if action.kind == ActionKind.KEY and action.keys and "win" in action.keys:
                    time.sleep(0.20)
                else:
                    time.sleep(0.08)
                step_idx += 1

        except Exception:
            raise
        finally:
            streamer.stop()
            try:
                self.computer.emergency_release_hardware()
            except Exception:
                pass

        if not session.is_complete and step_idx > max_steps:
            session.finish(f"Maximum step limit ({max_steps}) reached.")
            speak_feedback("Autopilot reached maximum step limit.", enabled=self.enable_voice)

        metrics = session.get_metrics()
        logger.info(f"Session finished: {metrics}")
        return session


def main() -> None:
    """CLI Entrypoint for the Ace Autopilot."""
    parser = argparse.ArgumentParser(
        description="AVCS Ace-Control-Medium Autonomous Visual GUI Autopilot."
    )
    parser.add_argument("--goal", type=str, required=True, help="User goal to execute on the desktop.")
    parser.add_argument("--endpoint", type=str, default=None, help="Custom LLM API endpoint URL.")
    parser.add_argument("--model", type=str, default=None, help="Frontier model identifier (e.g. gpt-6-astra, fable-5.5).")
    parser.add_argument("--api-key", type=str, default=None, help="API key for custom LLM provider.")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum execution steps before aborting.")
    parser.add_argument("--format", type=str, default="jpeg", choices=["webp", "jpeg", "png"], help="Screen capture compression format.")
    parser.add_argument("--protocol", type=str, default=None, choices=["openai", "anthropic", "gemini", "auto"], help="API wire protocol.")
    parser.add_argument("--timeout", type=float, default=None, help="HTTP request timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Observe and plan without dispatching hardware mouse/keyboard events.")
    parser.add_argument("--no-voice", action="store_true", help="Disable natural voice speech feedback.")

    args = parser.parse_args()

    client = CustomLLMClient(
        endpoint=args.endpoint,
        model=args.model,
        api_key=args.api_key,
        protocol=args.protocol,
        timeout=args.timeout,
    )
    agent = AceAgent(client=client)
    computer = WindowsComputer()
    runner = AceRunner(agent=agent, computer=computer, enable_voice=not args.no_voice)

    try:
        runner.run(
            goal=args.goal,
            max_steps=args.max_steps,
            image_format=args.format,
            dry_run=args.dry_run,
        )
    except SafetyBreakoutError as e:
        logger.warning(f"EMERGENCY BREAKOUT: {e}")
        speak_feedback("Autopilot emergency breakout triggered by user.")
        sys.exit(2)
    except KeyboardInterrupt:
        logger.info("Autopilot stopped by user via KeyboardInterrupt.")
        sys.exit(0)


if __name__ == "__main__":
    main()

