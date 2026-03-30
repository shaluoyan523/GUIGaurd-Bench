"""
Semantic Consistency Grader for AI Agent Plans

This module provides a grader to evaluate the semantic similarity and consistency
between two AI agent action plans, using a 5-point numerical scale.
"""

from textwrap import dedent

from planner_eval.judges.base import BaseJudge, Judgment


class SemanticConsistencyGrader(BaseJudge):
    """
    A grader that evaluates the semantic consistency between two AI agent plans
    using a 5-point scale (0-4).
    
    This grader is specifically designed for evaluating whether two AI agent action
    plans express the same intent and achieve the same goal, even if the exact
    wording or implementation details differ.
    
    Score Scale:
    -----------
    [0] Completely Inconsistent: Plans describe entirely different actions or goals
        (e.g., clicking vs. typing, or targeting completely different elements).
    
    [1] Minimally Consistent: Plans share some superficial similarity but have
        different intents or targets (e.g., both involve clicking but on different
        elements with different purposes).
    
    [2] Partially Consistent: Plans have the same general intent but differ in
        approach or implementation (e.g., both try to open a file but use different
        methods like clicking the file name vs. clicking an "Open" button).
    
    [3] Mostly Consistent: Plans are semantically similar with only minor differences
        in wording or coordinates (e.g., same action type and target, slightly
        different descriptions or nearby coordinates).
    
    [4] Fully Consistent: Plans are semantically identical, expressing the exact
        same action, target, and intent (minor wording variations are acceptable).
    
    Usage Example:
    -------------
    >>> grader = SemanticConsistencyGrader(model='openai/gpt-4-turbo')
    >>> judgment = grader.judge(
    ...     input="Task context (optional)",
    ...     output="Replay plan to evaluate",
    ...     expected="Ground truth plan"
    ... )
    >>> print(f"Score: {judgment.score}/4")
    >>> print(f"Reasoning: {judgment.reasoning}")
    """
    
    def judge(
        self,
        input: str = None,
        output: str = None,
        expected: str = None,
    ) -> Judgment:
        """
        Evaluate the semantic consistency between two AI agent plans.
        
        Parameters:
        ----------
        input: str (optional)
            Task description or context that both plans are trying to accomplish.
        output: str
            The replay plan to be evaluated (the plan generated during replay).
        expected: str
            The ground truth plan (the original plan from the baseline execution).
        
        Returns:
        -------
        Judgment
            A judgment object containing:
            - score (int): 0-4 indicating the level of semantic consistency
            - reasoning (str): Detailed explanation of the score
            - score_type (str): "numerical"
        """
        system_prompt = "You are an expert at evaluating semantic consistency of AI agent action plans."
        
        task_context = f"\nTask Context: {input}\n" if input else ""
        
        user_prompt = dedent(
            f"""
            Assess the semantic consistency between the GROUND TRUTH PLAN and the REPLAY PLAN on a five-point scale:
            
            [0] Completely Inconsistent: Plans describe entirely different actions or goals (e.g., clicking vs. typing, or targeting completely different elements).
            
            [1] Minimally Consistent: Plans share some superficial similarity but have different intents or targets (e.g., both involve clicking but on different elements with different purposes).
            
            [2] Partially Consistent: Plans have the same general intent but differ in approach or implementation (e.g., both try to open a file but use different methods like clicking the file name vs. clicking an "Open" button).
            
            [3] Mostly Consistent: Plans are semantically similar with only minor differences in wording or coordinates (e.g., same action type and target, slightly different descriptions or nearby coordinates).
            
            [4] Fully Consistent: Plans are semantically identical, expressing the exact same action, target, and intent (minor wording variations are acceptable).
            
            Consider these criteria:
            - Action Type: Do both plans involve the same type of action (click, type, navigate, etc.)?
            - Target Element: Do both plans target the same or equivalent UI elements?
            - Intent & Goal: Do both plans express the same underlying intent and goal?
            - Implementation: Are the approaches fundamentally the same, even if details differ?
            {task_context}
            Ground Truth Plan:
            {expected}
            
            Replay Plan:
            {output}
            
            Semantic Consistency Score (0-4):
            """
        )
        
        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        
        return Judgment(reasoning=reasoning, score=score, score_type="numerical")
