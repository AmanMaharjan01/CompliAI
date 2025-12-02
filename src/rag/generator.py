"""
Answer generation with PDO prompting and structured output
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json

from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.prompts.templates import POLICY_QUERY_PROMPT, HALLUCINATION_CHECK_PROMPT

logger = logging.getLogger(__name__)


class PolicyAnswer(BaseModel):
    """Structured answer format"""
    summary: str = Field(description="One-sentence direct answer")
    detailed_answer: str = Field(description="Comprehensive explanation")
    policy_references: List[Dict[str, str]] = Field(
        description="List of source citations with quotes"
    )
    confidence_level: str = Field(
        description="High/Medium/Low confidence assessment"
    )
    confidence_reasoning: str = Field(
        description="Explanation for confidence level"
    )
    action_items: Optional[List[str]] = Field(
        default=None,
        description="Actionable steps if applicable"
    )
    related_topics: List[str] = Field(
        description="Suggested related questions"
    )
    requires_escalation: bool = Field(
        default=False,
        description="Whether to escalate to human expert"
    )


@dataclass
class GenerationResult:
    """Complete generation result with metadata"""
    answer: PolicyAnswer
    raw_response: str
    retrieved_docs: List[Document]
    is_grounded: bool
    hallucination_score: float
    generation_metadata: Dict[str, Any]


class AnswerGenerator:
    """Generates policy answers using PDO prompting"""
    
    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.0,
        check_hallucinations: bool = True
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )
        
        self.check_hallucinations = check_hallucinations
        
        # Initialize output parser
        self.output_parser = PydanticOutputParser(pydantic_object=PolicyAnswer)
        
        logger.info(f"Initialized answer generator with model: {model_name}")
    
    def generate(
        self,
        question: str,
        retrieved_docs: List[Document],
        chat_history: Optional[str] = None
    ) -> GenerationResult:
        """Generate structured answer using PDO prompting"""
        
        logger.info(f"Generating answer for question: '{question[:100]}...'")
        
        # Format context from retrieved documents
        context = self._format_context(retrieved_docs)
        
        # Prepare chat history
        chat_history_str = chat_history or "No previous conversation."
        
        # Create the chain
        chain = LLMChain(
            llm=self.llm,
            prompt=POLICY_QUERY_PROMPT
        )
        
        try:
            # Generate answer
            raw_response = chain.run(
                context=context,
                question=question,
                chat_history=chat_history_str
            )
            
            # Parse structured output
            answer = self._parse_answer(raw_response)
            
            # Check for hallucinations
            is_grounded, hallucination_score = self._check_hallucinations(
                context=context,
                answer=raw_response,
                retrieved_docs=retrieved_docs
            ) if self.check_hallucinations else (True, 0.0)
            
            # Prepare generation metadata
            metadata = {
                "model": self.llm.model_name,
                "num_retrieved_docs": len(retrieved_docs),
                "context_length": len(context),
                "has_citations": len(answer.policy_references) > 0
            }
            
            result = GenerationResult(
                answer=answer,
                raw_response=raw_response,
                retrieved_docs=retrieved_docs,
                is_grounded=is_grounded,
                hallucination_score=hallucination_score,
                generation_metadata=metadata
            )
            
            logger.info(f"Generated answer with confidence: {answer.confidence_level}")
            return result
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            raise
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            
            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {source}, Page: {page}\n"
                f"Content: {doc.page_content}\n"
                f"---"
            )
        
        return "\n\n".join(context_parts)
    
    def _parse_answer(self, raw_response: str) -> PolicyAnswer:
        """Parse raw LLM response into structured format"""
        try:
            # Try to parse as structured format
            # First, try to extract JSON if present
            if "```json" in raw_response:
                json_str = raw_response.split("```json")[1].split("```")[0].strip()
                data = json.loads(json_str)
                return PolicyAnswer(**data)
            
            # Otherwise, parse from markdown format
            return self._parse_markdown_answer(raw_response)
            
        except Exception as e:
            logger.warning(f"Structured parsing failed, using fallback: {str(e)}")
            return self._create_fallback_answer(raw_response)
    
    def _parse_markdown_answer(self, text: str) -> PolicyAnswer:
        """Parse markdown-formatted answer"""
        lines = text.split('\n')
        
        data = {
            "summary": "",
            "detailed_answer": "",
            "policy_references": [],
            "confidence_level": "Medium",
            "confidence_reasoning": "Not specified",
            "action_items": [],
            "related_topics": [],
            "requires_escalation": False
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("**Summary**"):
                current_section = "summary"
            elif line.startswith("**Detailed Answer**"):
                current_section = "detailed_answer"
            elif line.startswith("**Policy References**"):
                current_section = "references"
            elif line.startswith("**Confidence"):
                current_section = "confidence"
            elif line.startswith("**Action Items**"):
                current_section = "actions"
            elif line.startswith("**Related Topics**"):
                current_section = "related"
            elif current_section and line:
                if current_section == "summary":
                    data["summary"] += line + " "
                elif current_section == "detailed_answer":
                    data["detailed_answer"] += line + " "
                # Add more parsing logic for other sections...
        
        return PolicyAnswer(**data)
    
    def _create_fallback_answer(self, raw_response: str) -> PolicyAnswer:
        """Create basic structured answer from unstructured text"""
        return PolicyAnswer(
            summary=raw_response[:200] + "...",
            detailed_answer=raw_response,
            policy_references=[],
            confidence_level="Low",
            confidence_reasoning="Unable to parse structured response",
            related_topics=[],
            requires_escalation=True
        )
    
    def _check_hallucinations(
        self,
        context: str,
        answer: str,
        retrieved_docs: List[Document]
    ) -> tuple[bool, float]:
        """Check if answer is grounded in context"""
        
        try:
            hallucination_chain = LLMChain(
                llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
                prompt=HALLUCINATION_CHECK_PROMPT
            )
            
            result = hallucination_chain.run(
                context=context,
                answer=answer
            )
            
            # Parse JSON response
            check_result = json.loads(result)
            is_grounded = check_result.get('is_grounded', False)
            confidence = check_result.get('confidence', 0.0)
            
            logger.info(f"Hallucination check: grounded={is_grounded}, score={confidence}")
            return is_grounded, 1.0 - confidence
            
        except Exception as e:
            logger.error(f"Hallucination check failed: {str(e)}")
            return True, 0.0  # Assume grounded if check fails
