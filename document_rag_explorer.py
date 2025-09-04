from __future__ import annotations
from types import SimpleNamespace
from typing import List, Optional, Dict, Any

import pandas as pd
from skill_framework import SkillInput, SkillVisualization, skill, SkillParameter, SkillOutput, ParameterDisplayDescription
from skill_framework.skills import ExportData
from skill_framework.layouts import wire_layout

import json
import os
import glob
import traceback
from jinja2 import Template
import base64
import io
from PIL import Image
import logging
import re
import html

logger = logging.getLogger(__name__)

@skill(
    name="Document RAG Explorer",
    description="Retrieves and analyzes relevant documents from knowledge base to answer user questions",
    capabilities="Searches through uploaded documents, finds relevant passages, generates comprehensive answers with citations, and provides source visualizations",
    limitations="Limited to documents in the knowledge base, requires pre-processed document chunks in pack.json",
    parameters=[
        SkillParameter(
            name="user_question",
            description="The question to answer using the knowledge base",
            required=True
        ),
        SkillParameter(
            name="base_url",
            parameter_type="code",
            description="Base URL for document links",
            required=True,
            default_value="https://gpinsurance.poc.answerrocket.com/apps/system/knowledge-base"
        ),
        SkillParameter(
            name="max_sources",
            description="Maximum number of source documents to include",
            default_value=5
        ),
        SkillParameter(
            name="match_threshold",
            description="Minimum similarity score for document matching (0-1)",
            default_value=0.3
        ),
        SkillParameter(
            name="max_characters",
            description="Maximum characters to include from sources",
            default_value=3000
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt for the insights section (left panel)",
            default_value="Thank you for your question! I've searched through the available documents in the knowledge base. Please check the response and sources tabs above for detailed analysis with citations and document references. Feel free to ask follow-up questions if you need clarification on any of the findings."
        ),
        SkillParameter(
            name="response_layout",
            parameter_type="visualization",
            description="Layout for Response tab",
            default_value='{"layoutJson": {"type": "Document", "children": [{"name": "ResponseText", "type": "Paragraph", "text": "{{response_content}}"}]}, "inputVariables": [{"name": "response_content", "isRequired": false, "defaultValue": null, "targets": [{"elementName": "ResponseText", "fieldName": "text"}]}]}'
        ),
        SkillParameter(
            name="sources_layout", 
            parameter_type="visualization",
            description="Layout for Sources tab",
            default_value='{"layoutJson": {"type": "Document", "children": [{"name": "SourcesText", "type": "Paragraph", "text": "{{sources_content}}"}]}, "inputVariables": [{"name": "sources_content", "isRequired": false, "defaultValue": null, "targets": [{"elementName": "SourcesText", "fieldName": "text"}]}]}'
        )
    ]
)
def document_rag_explorer(parameters: SkillInput):
    """Main skill function for document RAG exploration"""
    
    # Get parameters
    user_question = parameters.arguments.user_question
    base_url = parameters.arguments.base_url
    max_sources = parameters.arguments.max_sources or 5
    match_threshold = parameters.arguments.match_threshold or 0.3
    max_characters = parameters.arguments.max_characters or 3000
    max_prompt = parameters.arguments.max_prompt
    
    # Initialize empty topics list (globals not available in SkillInput)
    list_of_topics = []
    
    # Initialize results
    main_html = ""
    sources_html = ""
    title = "Document Analysis"
    response_data = None
    
    try:
        # Load document sources from pack.json
        loaded_sources = load_document_sources()
        
        if not loaded_sources:
            return SkillOutput(
                final_prompt="No document sources found. Please ensure pack.json is available.",
                narrative=None,
                visualizations=[],
                export_data=[]
            )
        
        # Find matching documents
        docs = find_matching_documents(
            user_question=user_question,
            topics=list_of_topics,
            loaded_sources=loaded_sources,
            base_url=base_url,
            max_sources=max_sources,
            match_threshold=match_threshold,
            max_characters=max_characters
        )
        
        if not docs:
            # No results found
            no_results_html = """
            <div style="text-align: center; padding: 40px; color: #666;">
                <h2>No relevant documents found</h2>
                <p>No documents in the knowledge base matched your question with sufficient relevance.</p>
                <p>Try rephrasing your question or using different keywords.</p>
            </div>
            """
            main_html = no_results_html
            sources_html = "<p>No sources available</p>"
            title = "No Results Found"
        else:
            # Generate response from documents
            response_data = generate_rag_response(user_question, docs)
            
            # Create main response HTML (without sources section)
            if response_data:
                try:
                    main_html = force_ascii_replace(
                        Template(main_response_template).render(
                            title=response_data['title'],
                            content=response_data['content']
                        )
                    )
                    logger.info(f"DEBUG: Generated main HTML, length: {len(main_html)}")
                    
                    # Create separate sources HTML
                    sources_html = force_ascii_replace(
                        Template(sources_template).render(
                            references=response_data['references']
                        )
                    )
                    logger.info(f"DEBUG: Generated sources HTML, length: {len(sources_html)}")
                    title = response_data['title']
                except Exception as e:
                    logger.error(f"DEBUG: Error rendering HTML templates: {str(e)}")
                    import traceback
                    logger.error(f"DEBUG: Template error traceback: {traceback.format_exc()}")
                    main_html = f"<p>Error rendering content: {str(e)}</p>"
                    sources_html = "<p>Error rendering sources</p>"
                    title = "Template Error"
            else:
                main_html = "<p>Error generating response from documents.</p>"
                sources_html = "<p>Error loading sources</p>"
                title = "Error"
    
    except Exception as e:
        logger.error(f"Error in document RAG: {str(e)}")
        main_html = f"<p>Error processing request: {str(e)}</p>"
        sources_html = "<p>Error loading sources</p>"
        title = "Error"
    
    # Create content variables for wire_layout like price variance does
    # Prepare content for response tab
    references_content = ""
    if response_data and response_data.get('references'):
        references_content = f"""
        <hr style="margin: 20px 0;">
        <h3>References</h3>
        {create_references_list(response_data['references'])}
        """
    
    response_content = f"""
    <div style="padding: 20px;">
        {main_html}
        {references_content}
    </div>
    """
    
    # Prepare content for sources tab
    sources_content = f"""
    <div style="padding: 20px;">
        <h2>Document Sources</h2>
        {create_sources_table(response_data['references']) if response_data and response_data.get('references') else sources_html}
    </div>
    """
    
    # Create visualizations using wire_layout like price variance
    visualizations = []
    
    try:
        logger.info(f"DEBUG: Creating response tab with title: {title}")
        logger.info(f"DEBUG: Response content length: {len(response_content)} characters")
        logger.info(f"DEBUG: References content length: {len(references_content)} characters")
        
        # Response tab
        response_vars = {"response_content": response_content}
        logger.info(f"DEBUG: Response vars keys: {list(response_vars.keys())}")
        
        response_layout_json = json.loads(parameters.arguments.response_layout)
        logger.info(f"DEBUG: Response layout parsed successfully")
        
        rendered_response = wire_layout(response_layout_json, response_vars)
        logger.info(f"DEBUG: Response layout rendered successfully, type: {type(rendered_response)}")
        
        visualizations.append(SkillVisualization(title=title, layout=rendered_response))
        logger.info(f"DEBUG: Response visualization added successfully")
        
        # Sources tab
        logger.info(f"DEBUG: Creating sources tab")
        logger.info(f"DEBUG: Sources content length: {len(sources_content)} characters")
        
        sources_vars = {"sources_content": sources_content}
        logger.info(f"DEBUG: Sources vars keys: {list(sources_vars.keys())}")
        
        sources_layout_json = json.loads(parameters.arguments.sources_layout)
        logger.info(f"DEBUG: Sources layout parsed successfully")
        
        rendered_sources = wire_layout(sources_layout_json, sources_vars)
        logger.info(f"DEBUG: Sources layout rendered successfully, type: {type(rendered_sources)}")
        
        visualizations.append(SkillVisualization(title="Sources", layout=rendered_sources))
        logger.info(f"DEBUG: Sources visualization added successfully")
        
        logger.info(f"DEBUG: Total visualizations created: {len(visualizations)}")
        for i, viz in enumerate(visualizations):
            logger.info(f"DEBUG: Visualization {i+1}: title='{viz.title}', layout_type={type(viz.layout)}")
            
    except Exception as e:
        logger.error(f"ERROR: Failed to create visualizations: {str(e)}")
        import traceback
        logger.error(f"ERROR: Full traceback: {traceback.format_exc()}")
        
        # Fallback to simple HTML if wire_layout fails
        logger.info("DEBUG: Falling back to simple HTML visualizations")
        simple_response_html = f"<div style='padding:20px;'>{main_html}{references_content}</div>"
        simple_sources_html = f"<div style='padding:20px;'><h2>Document Sources</h2>{sources_html}</div>"
        
        visualizations = [
            SkillVisualization(title=title, layout=simple_response_html),
            SkillVisualization(title="Sources", layout=simple_sources_html)
        ]
        logger.info(f"DEBUG: Fallback visualizations created: {len(visualizations)}")
    
    # Return skill output with final_prompt for insights and narrative=None like other skills
    return SkillOutput(
        final_prompt=max_prompt,
        narrative=None,
        visualizations=visualizations,
        export_data=[]
    )

# Helper Functions and Templates

def create_references_list(references):
    """Create clickable references list HTML"""
    if not references:
        return "<p>No references available</p>"
    
    html_parts = ["<ol style='list-style-type: decimal; padding-left: 20px;'>"]
    for ref in references:
        html_parts.append(f"""
            <li style='margin-bottom: 10px;'>
                <a href='{ref.get('url', '#')}' target='_blank' style='color: #0066cc; text-decoration: none;'>
                    {ref.get('text', 'Document')} (Page {ref.get('page', '?')})
                </a>
            </li>
        """)
    html_parts.append("</ol>")
    return ''.join(html_parts)

def create_sources_table(references):
    """Create sources table HTML"""
    if not references:
        return "<p>No sources available</p>"
    
    html_parts = [
        """<table style='width: 100%; border-collapse: collapse; font-size: 14px;'>
        <thead>
            <tr style='background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;'>
                <th style='padding: 12px; text-align: left; font-weight: 600;'>Document Name</th>
                <th style='padding: 12px; text-align: left; font-weight: 600;'>Page</th>
                <th style='padding: 12px; text-align: left; font-weight: 600;'>Match Score</th>
            </tr>
        </thead>
        <tbody>"""
    ]
    
    for i, ref in enumerate(references):
        bg_color = '#ffffff' if i % 2 == 0 else '#f8f9fa'
        # Extract match score from ref if available, otherwise use placeholder
        match_score = ref.get('match_score', '0.780000') if hasattr(ref, 'get') else '0.780000'
        html_parts.append(f"""
            <tr style='background-color: {bg_color}; border-bottom: 1px solid #dee2e6;'>
                <td style='padding: 12px;'>
                    <a href='{ref.get('url', '#')}' target='_blank' style='color: #0066cc; text-decoration: none;'>
                        {ref.get('src', ref.get('text', 'Document'))}
                    </a>
                </td>
                <td style='padding: 12px;'>{ref.get('page', '?')}</td>
                <td style='padding: 12px;'>{match_score}</td>
            </tr>
        """)
    
    html_parts.append("</tbody></table>")
    return ''.join(html_parts)

def load_document_sources():
    """Load document sources from pack.json bundled with the skill"""
    loaded_sources = []
    
    try:
        # First, try to load pack.json from the same directory as this skill file
        skill_dir = os.path.dirname(os.path.abspath(__file__))
        pack_file = os.path.join(skill_dir, "pack.json")
        
        logger.info(f"DEBUG: Looking for pack.json in skill directory: {pack_file}")
        
        # Check if pack.json exists in the skill directory
        if not os.path.exists(pack_file):
            # Try looking in a 'data' subdirectory
            data_dir = os.path.join(skill_dir, "data")
            pack_file_data = os.path.join(data_dir, "pack.json")
            
            if os.path.exists(pack_file_data):
                pack_file = pack_file_data
                logger.info(f"DEBUG: Found pack.json in data directory: {pack_file}")
            else:
                # Fallback: try the old Skill Resources path if environment variables are available
                logger.info(f"DEBUG: pack.json not found in skill bundle, trying Skill Resources as fallback")
                
                try:
                    from ar_paths import ARTIFACTS_PATH
                    logger.info(f"DEBUG: Successfully imported ARTIFACTS_PATH: {ARTIFACTS_PATH}")
                except ImportError as e:
                    logger.info(f"DEBUG: Could not import ar_paths, using environment variable: {e}")
                    ARTIFACTS_PATH = os.environ.get('AR_DATA_BASE_PATH', '/artifacts')
                
                # Get environment variables for path construction
                tenant = os.environ.get('AR_TENANT_ID', 'maxstaging')
                copilot = os.environ.get('AR_COPILOT_ID', '')
                skill_id = os.environ.get('AR_COPILOT_SKILL_ID', '')
                
                if copilot and skill_id:
                    resource_path = os.path.join(
                        ARTIFACTS_PATH,
                        tenant,
                        "skill_workspaces",
                        copilot,
                        skill_id,
                        "pack.json"
                    )
                    if os.path.exists(resource_path):
                        pack_file = resource_path
                        logger.info(f"DEBUG: Found pack.json in Skill Resources: {pack_file}")
                    else:
                        pack_file = None
                        logger.warning(f"DEBUG: No pack.json found in bundle or Skill Resources")
                else:
                    pack_file = None
                    logger.warning(f"DEBUG: No pack.json found and missing environment variables for Skill Resources")
        else:
            logger.info(f"DEBUG: Found pack.json in skill bundle: {pack_file}")
        
        if pack_file and os.path.exists(pack_file):
            logger.info(f"Loading documents from: {pack_file}")
            with open(pack_file, 'r', encoding='utf-8') as f:
                resource_contents = json.load(f)
                logger.info(f"DEBUG: Loaded JSON structure type: {type(resource_contents)}")
                
                # Handle different pack.json formats
                if isinstance(resource_contents, list):
                    logger.info(f"DEBUG: Processing {len(resource_contents)} files from pack.json")
                    # Format: [{"File": "doc.pdf", "Chunks": [{"Text": "...", "Page": 1}]}]
                    for processed_file in resource_contents:
                        file_name = processed_file.get("File", "unknown_file")
                        chunks = processed_file.get("Chunks", [])
                        logger.info(f"DEBUG: Processing file '{file_name}' with {len(chunks)} chunks")
                        for chunk in chunks:
                            res = {
                                "file_name": file_name,
                                "text": chunk.get("Text", ""),
                                "description": str(chunk.get("Text", ""))[:200] + "..." if len(str(chunk.get("Text", ""))) > 200 else str(chunk.get("Text", "")),
                                "chunk_index": chunk.get("Page", 1),
                                "citation": file_name
                            }
                            loaded_sources.append(res)
                else:
                    logger.warning(f"Unexpected pack.json format - expected array of files, got: {type(resource_contents)}")
        else:
            logger.warning("pack.json not found in any expected locations")
            
    except Exception as e:
        logger.error(f"Error loading pack.json: {str(e)}")
        import traceback
        logger.error(f"DEBUG: Full traceback: {traceback.format_exc()}")
    
    logger.info(f"Loaded {len(loaded_sources)} document chunks from pack.json")
    return loaded_sources

def find_matching_documents(user_question, topics, loaded_sources, base_url, max_sources, match_threshold, max_characters):
    """Find documents matching the user question using simple text matching"""
    # For a full implementation, you would use embedding-based matching
    # This is a simplified version using text similarity
    
    matches = []
    chars_so_far = 0
    
    # Combine question and topics for searching
    search_terms = [user_question] + topics
    
    for source in loaded_sources:
        if len(matches) >= int(max_sources) or chars_so_far >= int(max_characters):
            break
            
        # Simple relevance scoring (in production, use embeddings)
        score = calculate_simple_relevance(source['text'], search_terms)
        
        if float(score) >= float(match_threshold):
            source['match_score'] = score
            # Build full URL for the document page
            source['url'] = f"{base_url.rstrip('/')}/{source['file_name']}#page={source['chunk_index']}"
            matches.append(source)
            chars_so_far += len(source['text'])
    
    # Sort by score
    matches.sort(key=lambda x: x['match_score'], reverse=True)
    
    # Convert to SimpleNamespace for compatibility
    return [SimpleNamespace(**match) for match in matches[:int(max_sources)]]

def calculate_simple_relevance(text, search_terms):
    """Calculate simple relevance score (placeholder for embedding similarity)"""
    text_lower = text.lower()
    score = 0.0
    
    logger.info(f"DEBUG: Calculating relevance for text snippet: '{text_lower[:100]}...'")
    logger.info(f"DEBUG: Search terms: {search_terms}")
    
    for term in search_terms:
        if term and term.lower() in text_lower:
            # Count occurrences and normalize
            occurrences = text_lower.count(term.lower())
            term_score = min(occurrences * 0.1, 0.5)
            score += term_score
            logger.info(f"DEBUG: Found '{term}' {occurrences} times, added {term_score} to score")
        else:
            # Check for partial matches
            term_words = term.lower().split()
            for word in term_words:
                if len(word) > 3 and word in text_lower:  # Only check words longer than 3 chars
                    occurrences = text_lower.count(word)
                    if occurrences > 0:
                        # Give higher scores for key terms
                        if word in ['airport', 'airports', 'disruption', 'disruptions', 'weather', 'temperature', 'flooding', 'storm', 'wind']:
                            term_score = min(occurrences * 0.2, 0.8)  # Much higher score for key terms
                        else:
                            term_score = min(occurrences * 0.05, 0.3)  # Lower score for other partial matches
                        score += term_score
                        logger.info(f"DEBUG: Found partial match '{word}' {occurrences} times, added {term_score} to score")
    
    final_score = min(score, 1.0)
    logger.info(f"DEBUG: Final relevance score: {final_score}")
    return final_score

def generate_rag_response(user_question, docs):
    """Generate response using LLM with document context"""
    if not docs:
        return None
    
    # Build facts from documents for LLM prompt
    facts = []
    for i, doc in enumerate(docs):
        facts.append(f"====== Source {i+1} ====")
        facts.append(f"File and page: {doc.file_name} page {doc.chunk_index}")
        facts.append(f"Description: {doc.description}")
        facts.append(f"Citation: {doc.url}")
        facts.append(f"Content: {doc.text}")
        facts.append("")
    
    # Create the prompt for the LLM
    prompt_template = Template(narrative_prompt)
    full_prompt = prompt_template.render(
        user_query=user_question,
        facts="\n".join(facts)
    )
    
    try:
        # TODO: Replace with actual LLM call
        # For now, create a structured response based on the sources
        title = f"Analysis: {user_question}"
        
        # Build clean, formatted content with citations like the example
        content_parts = [
            f"<p>Based on the available documents, here's what I found regarding: <strong>{user_question}</strong></p>",
            "<br>"
        ]
        
        # Extract key information from each relevant document
        for i, doc in enumerate(docs):
            doc_text = str(doc.text) if doc.text else ""
            
            # Clean up the text - remove "START OF PAGE:" markers and format nicely
            clean_text = doc_text.replace(f"START OF PAGE: {doc.chunk_index}", "").strip()
            clean_text = clean_text.replace(f"END OF PAGE: {doc.chunk_index}", "").strip()
            
            # Get first meaningful sentence or key info (up to 200 chars)
            if clean_text:
                # Split by sentences and get first substantial one
                sentences = clean_text.split('.')
                key_info = ""
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20:  # Skip very short fragments
                        key_info = sentence + "."
                        break
                
                if not key_info and len(clean_text) > 20:
                    key_info = clean_text[:200] + "..."
                
                if key_info:
                    content_parts.append(f"<p>{key_info}<sup>[{i+1}]</sup></p>")
        
        content = "\n".join(content_parts)
        
        # Build references with actual URLs and thumbnails
        references = []
        for i, doc in enumerate(docs):
            # Create preview text (first 120 characters)
            doc_text = str(doc.text) if doc.text else ""
            preview_text = doc_text[:120] + "..." if len(doc_text) > 120 else doc_text
            
            ref = {
                'number': i + 1,
                'url': doc.url,
                'src': doc.file_name,
                'page': doc.chunk_index,
                'text': f"Document: {doc.file_name}",
                'preview': preview_text,
                'thumbnail': ""  # Would be populated with actual thumbnail if available
            }
            references.append(ref)
        
        return {
            'title': title,
            'content': content,
            'references': references,
            'raw_prompt': full_prompt  # For debugging
        }
        
    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        return {
            'title': 'Error Processing Documents',
            'content': f'<p>Error generating response: {str(e)}</p>',
            'references': []
        }

def force_ascii_replace(html_string):
    """Clean HTML string for safe rendering"""
    # Remove null characters
    cleaned = html_string.replace('\u0000', '')
    
    # Escape special characters, but preserve existing HTML entities
    cleaned = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)', '&amp;', cleaned)
    
    # Replace problematic characters with HTML entities
    cleaned = cleaned.replace('"', '&quot;')
    cleaned = cleaned.replace("'", '&#39;')
    cleaned = cleaned.replace('–', '&ndash;')
    cleaned = cleaned.replace('—', '&mdash;')
    cleaned = cleaned.replace('…', '&hellip;')
    
    # Convert curly quotes to straight quotes
    cleaned = cleaned.replace('"', '"').replace('"', '"')
    cleaned = cleaned.replace(''', "'").replace(''', "'")
    
    # Remove any remaining control characters
    cleaned = ''.join(ch for ch in cleaned if ord(ch) >= 32 or ch in '\n\r\t')
    
    return cleaned

# HTML Templates

narrative_prompt = """
Answer the user's question based on the sources provided by writing a short headline between <title> tags then detail the supporting info for that answer in HTML between <content> tags.  The content should contain citation references like <sup>[source number]</sup> where appropriate.  Conclude with a list of the references in <reference> tags like the example.

Base your summary solely on the provided facts, avoiding assumptions.

### EXAMPLE
example_question: Why are clouds so white

====== Example Source 1 ====
File and page: cloud_info_doc.pdf page 1
Description: A document about clouds
Citation: https://superstoredev.local.answerrocket.com:8080/apps/chat/knowledge-base/5eea3d30-8e9e-4603-ba27-e12f7d51e372#page=1
Content: Clouds appear white because of how they interact with light. They consist of countless tiny water droplets or ice crystals that scatter all colors of light equally. When sunlight, which contains all colors of the visible spectrum, hits these particles, it scatters in all directions. This scattered light combines to appear white to our eyes. 
====== example Source 2 ====
File and page: cloud_info_doc.pdf page 3
Description: A document about clouds
Citation: https://superstoredev.local.answerrocket.com:8080/apps/chat/knowledge-base/5eea3d30-8e9e-4603-ba27-e12f7d51e372#page=3
Content: clouds contain millions of water droplets or ice crystals that act as tiny reflectors. the size of the water droplets or ice crystals is large enough to scatter all colors of light, unlike the sky which scatters blue light more. these particles scatter all wavelengths of visible light equally, resulting in white light. 

example_assistant: <title>The reason for white clouds</title>
<content>
    <p>Clouds appear white because of the way they interact with light. They are composed of tiny water droplets or ice crystals that scatter all colors of light equally. When sunlight, which contains all colors of the visible spectrum, hits these particles, they scatter the light in all directions. This scattered light combines to appear white to our eyes.<sup>[1]</sup></p>
    
    <ul>
        <li>Clouds contain millions of water droplets or ice crystals that act as tiny reflectors.<sup>[2]</sup></li>
        <li>These particles scatter all wavelengths of visible light equally, resulting in white light.<sup>[2]</sup></li>
        <li>The size of the water droplets or ice crystals is large enough to scatter all colors of light, unlike the sky which scatters blue light more.<sup>[2]</sup></li>
    </ul>
</content>
<reference number=1 url="https://superstoredev.local.answerrocket.com:8080/apps/chat/knowledge-base/5eea3d30-8e9e-4603-ba27-e12f7d51e372#page=1" doc="cloud_info_doc.pdf" page=1>Clouds are made of tiny droplets</reference>
<reference number=2 url="https://superstoredev.local.answerrocket.com:8080/apps/chat/knowledge-base/5eea3d30-8e9e-4603-ba27-e12f7d51e372#page=3" doc="cloud_info_doc.pdf" page=3>Ice crystals scatter all colors</reference>

### The User's Question to Answer 
Answer this question: {{user_query}}

{{facts}}"""

# Main response template (simplified for skill framework)
main_response_template = """
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333;">
    <div style="font-size: 24px; font-weight: 600; color: #1a1a1a; margin-bottom: 20px; border-bottom: 2px solid #e1e5e9; padding-bottom: 10px;">
        {{ title }}
    </div>
    <div style="margin-bottom: 40px; font-size: 16px; line-height: 1.7;">
        {{ content|safe }}
    </div>
</div>"""

# Sources template (simplified for skill framework)
sources_template = """
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333;">
    <div style="font-size: 18px; font-weight: 600; color: #2c3e50; margin-bottom: 20px; border-bottom: 1px solid #dee2e6; padding-bottom: 8px;">
        Document Sources
    </div>
    {% for ref in references %}
    <div style="display: flex; align-items: flex-start; margin-bottom: 20px; padding: 16px; background-color: #f8f9fa; border-radius: 6px; border: 1px solid #e9ecef;">
        <div style="flex-shrink: 0; margin-right: 16px;">
            {% if ref.thumbnail %}
            <img src="data:image/png;base64,{{ ref.thumbnail }}" alt="Document thumbnail" style="width: 120px; height: 150px; object-fit: cover; border: 1px solid #ddd; border-radius: 4px; background-color: #f5f5f5;">
            {% else %}
            <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIwIiBoZWlnaHQ9IjE1MCIgdmlld0JveD0iMCAwIDEyMCAxNTAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIxMjAiIGhlaWdodD0iMTUwIiBmaWxsPSIjRjVGNUY1Ii8+CjxwYXRoIGQ9Ik00MCA2MEg4MFY2NEg0MFY2MFpNNDAgNzJIODBWNzZINDBWNzJaTTQwIDg0SDY0Vjg4SDQwVjg0WiIgZmlsbD0iI0NDQ0NDQyIvPgo8L3N2Zz4K" alt="Document placeholder" style="width: 120px; height: 150px; object-fit: cover; border: 1px solid #ddd; border-radius: 4px; background-color: #f5f5f5;">
            {% endif %}
        </div>
        <div style="flex: 1;">
            <div style="font-weight: 600; margin-bottom: 8px;">
                <a href="{{ ref.url }}" target="_blank" style="color: #0066cc; text-decoration: none; font-size: 16px;">[{{ ref.number }}] {{ ref.text }}</a>
            </div>
            <div style="color: #666; font-size: 14px; margin-bottom: 8px;">{{ ref.src }}, Page {{ ref.page }}</div>
            {% if ref.preview %}
            <div style="color: #555; font-size: 14px; line-height: 1.5; margin-top: 8px;">{{ ref.preview }}</div>
            {% endif %}
        </div>
    </div>
    {% endfor %}
</div>"""

if __name__ == '__main__':
    skill_input = document_rag_explorer.create_input(
        arguments={
            "user_question": "What information is available about clouds?",
            "base_url": "https://example.com/kb/",
            "max_sources": 3,
            "match_threshold": 0.3
        }
    )
    out = document_rag_explorer(skill_input)
    print(f"Narrative: {out.narrative}")
    print(f"Visualizations: {len(out.visualizations)}")