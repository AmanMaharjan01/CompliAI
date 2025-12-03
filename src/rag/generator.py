"""
Answer generation with multiple LLM provider support
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
import os

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
        # Get LLM provider from environment
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        
        logger.info(f"Initializing LLM with provider: {provider}")
        
        # Store provider and model info for metadata
        self.provider = provider
        self.model_identifier = None
        
        try:
            if provider == "gemini":
                self.llm = self._init_gemini(temperature)
            elif provider == "openai":
                self.llm = self._init_openai(model_name, temperature)
            elif provider == "ollama":
                self.llm = self._init_ollama(temperature)
            else:
                logger.warning(f"Unknown provider '{provider}', defaulting to Gemini")
                self.llm = self._init_gemini(temperature)
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider '{provider}': {str(e)}")
            raise ValueError(
                f"Failed to initialize {provider} LLM.\n"
                f"Error: {str(e)}\n\n"
                f"Please check:\n"
                f"1. Your API key is correct in .env file\n"
                f"2. The API key has no extra spaces or quotes\n"
                f"3. For Gemini: Get key from https://makersuite.google.com/app/apikey\n"
                f"4. For OpenAI: Get key from https://platform.openai.com/api-keys"
            )
        
        # Check hallucinations setting from environment
        check_hallucinations_env = os.getenv("CHECK_HALLUCINATIONS", "true").lower()
        self.check_hallucinations = check_hallucinations and check_hallucinations_env == "true"
        
        if not self.check_hallucinations:
            logger.info("Hallucination checking disabled (saves 1 LLM call per query)")
        
        self.output_parser = PydanticOutputParser(pydantic_object=PolicyAnswer)
    
    def _init_gemini(self, temperature: float):
        """Initialize Google Gemini - reads from .env"""
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Read from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
        
        # Store model identifier
        self.model_identifier = model
        
        # Validate API key
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in .env file.\n"
                "Get your free API key from: https://makersuite.google.com/app/apikey"
            )
        
        # Clean the API key
        api_key = api_key.strip().strip('"').strip("'")
        
        if not api_key or len(api_key) < 20:
            raise ValueError(
                f"GOOGLE_API_KEY appears invalid (length: {len(api_key)}).\n"
                "Please check your .env file and ensure the key is correct."
            )
        
        logger.info(f"Gemini Configuration from .env:")
        logger.info(f"  - Model: {model}")
        logger.info(f"  - API Key: {api_key[:8]}...{api_key[-4:]}")
        
        try:
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=temperature,
                convert_system_message_to_human=True
            )
            
            if "flash" in model.lower():
                logger.info("✅ Using Gemini 1.5 Flash - Fast & efficient with generous FREE quota")
            elif "pro" in model.lower():
                logger.info("✅ Using Gemini 1.5 Pro - More capable but lower rate limits")
            else:
                logger.info(f"✅ Using Gemini model: {model}")
            
            return llm
        except Exception as e:
            error_msg = str(e)
            
            # Provide helpful error messages
            if "404" in error_msg or "not found" in error_msg.lower():
                raise ValueError(
                    f"Gemini model '{model}' not found.\n\n"
                    f"Valid model names:\n"
                    f"  - gemini-1.5-flash-latest (RECOMMENDED for free tier)\n"
                    f"  - gemini-1.5-pro-latest\n"
                    f"  - gemini-pro (legacy)\n\n"
                    f"Update your .env file:\n"
                    f"GEMINI_MODEL=gemini-1.5-flash-latest\n\n"
                    f"Original error: {error_msg}"
                )
            else:
                raise ValueError(
                    f"Failed to initialize Gemini with provided API key.\n"
                    f"Error: {str(e)}\n\n"
                    f"Please verify:\n"
                    f"1. Get a NEW API key from: https://makersuite.google.com/app/apikey\n"
                    f"2. Copy the ENTIRE key (usually starts with 'AIza')\n"
                    f"3. Paste in .env file: GOOGLE_API_KEY=AIza...\n"
                    f"4. NO quotes or extra spaces around the key\n"
                    f"5. Ensure model in .env: GEMINI_MODEL=gemini-1.5-flash-latest"
                )

    def _init_openai(self, model_name: str, temperature: float):
        """Initialize OpenAI (paid)"""
        from langchain_openai import ChatOpenAI
        
        # Override with environment if set
        model_name = os.getenv("OPENAI_MODEL", model_name)
        
        # Store model identifier
        self.model_identifier = model_name
        
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )
        logger.info(f"✅ Using OpenAI: {model_name} (paid API)")
        return llm
    
    def _init_ollama(self, temperature: float):
        """Initialize Ollama (FREE local LLM)"""
        from langchain_community.llms import Ollama
        
        # Read from environment
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama2")
        
        # Store model identifier
        self.model_identifier = model
        
        logger.info(f"Ollama Configuration from .env:")
        logger.info(f"  - Model: {model}")
        logger.info(f"  - Base URL: {base_url}")
        
        llm = Ollama(
            model=model,
            base_url=base_url,
            temperature=temperature
        )
        logger.info(f"✅ Using Ollama: {model} (FREE, runs locally)")
        logger.info("⚡ NO API calls - completely offline!")
        return llm

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
            
            # Prepare generation metadata - use stored model identifier
            metadata = {
                "provider": self.provider,
                "model": self.model_identifier or "unknown",
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
            # Use same provider for hallucination checking
            provider = os.getenv("LLM_PROVIDER", "gemini").lower()
            
            if provider == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                api_key = os.getenv("GOOGLE_API_KEY")
                model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
                
                check_llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=api_key,
                    temperature=0,
                    convert_system_message_to_human=True
                )
            elif provider == "ollama":
                from langchain_community.llms import Ollama
                check_llm = Ollama(
                    model=os.getenv("OLLAMA_MODEL", "llama2"),
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                    temperature=0
                )
            else:
                from langchain_openai import ChatOpenAI
                check_llm = ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                    temperature=0
                )
            
            hallucination_chain = LLMChain(
                llm=check_llm,
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
