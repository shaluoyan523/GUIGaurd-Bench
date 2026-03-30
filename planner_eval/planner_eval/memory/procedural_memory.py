import inspect
import textwrap


class PROCEDURAL_MEMORY:
    FORMATTING_FEEDBACK_PROMPT = textwrap.dedent(
        """
        Your previous response was not formatted correctly. Respond again and
        replace it completely. Do not mention this feedback explicitly.

        Please fix the following issues:
        FORMATTING_FEEDBACK
        """
    ).strip()

    @staticmethod
    def construct_simple_worker_procedural_memory(agent_class, skipped_actions):
        procedural_memory = textwrap.dedent(
            """\
            You are an expert UI planning agent working from prerecorded screenshots.
            You are responsible for solving the task: `TASK_DESCRIPTION`.
            You are working in CURRENT_OS.

            This environment is trajectory-only evaluation:
            - You DO NOT execute real desktop actions.
            - Your grounded action should still use the provided `agent.*` API.
            - The returned action is only used for symbolic evaluation.

            Your job on each step:
            1. Verify whether the previous action appears successful.
            2. Analyze the current screenshot carefully.
            3. Decide the single next action that best advances the task.
            4. Return exactly one grounded action.

            You are provided with:
            1. The current screenshot.
            2. The history of previous interactions.
            3. The following Agent API:
            class Agent:
            """
        )

        for attr_name in dir(agent_class):
            if attr_name in skipped_actions:
                continue
            attr = getattr(agent_class, attr_name)
            if callable(attr) and hasattr(attr, "is_agent_action"):
                signature = inspect.signature(attr)
                procedural_memory += f"""
    def {attr_name}{signature}:
    '''{attr.__doc__}'''
        """

        procedural_memory += textwrap.dedent(
            """

            Your response must use exactly this structure:
            (Previous action verification)
            Explain whether the previous action succeeded.

            (Screenshot Analysis)
            Describe the relevant UI state and visible cues.

            (Next Action)
            State the next action in natural language.

            (Grounded Action)
            Return exactly one Python code block with exactly one agent call, for example:
            ```python
            agent.click("Settings app icon", 1, "left")
            ```

            Rules:
            1. Only produce one grounded action.
            2. Do not include anything except Python code inside the code block.
            3. Use only the available `agent.*` methods.
            4. If the task is complete, return `agent.done()`.
            5. If the task is impossible or you are exhaustively stuck, return `agent.fail()`.
            6. Prefer concise, task-focused reasoning over long narration.
            """
        )
        return procedural_memory.strip()

    REFLECTION_ON_TRAJECTORY = textwrap.dedent(
        """
        You are an expert UI evaluation assistant reflecting on another agent's
        trajectory.

        You receive a task description and the recent trajectory. Your job is to
        emit one of three reflection cases:

        Case 1. The trajectory is not going according to plan.
        Case 2. The trajectory is going according to plan.
        Case 3. The task appears completed.

        Rules:
        - Do not prescribe a specific next action.
        - Focus on progress, cycles, and whether the observed screen state matches
          the task objective.
        - Keep Case 2 concise.
        """
    ).strip()
