"""
Reusable HTML component functions for consistent UI elements.
"""
import streamlit as st
from .icons import icon

def metric_card(label, value, subtitle="", center_align=False, value_size="2.6rem", extra_style=""):
    """
    Create a metric card with label, value, and optional subtitle.
    
    Args:
        label: Card label text
        value: Main value to display
        subtitle: Optional subtitle text
        center_align: Whether to center align the content
        value_size: Font size for the value
        extra_style: Additional inline styles
        
    Returns:
        HTML string for the metric card
    """
    align_style = "text-align:center;" if center_align else ""
    value_style = f"font-size: {value_size};" if value_size != "2.6rem" else ""
    
    subtitle_html = f'<div class="card-body" style="font-size: 0.78rem;">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class='intel-card' style='{align_style} {extra_style}'>
        <div class='card-label'>{label}</div>
        <div class='card-value' style='{value_style}'>{value}</div>
        {subtitle_html}
    </div>
    """
    
def capability_card(icon_svg, title, body):
    """
    Create a capability card with icon, title, and description.
    
    Args:
        icon_svg: SVG icon HTML string
        title: Card title
        body: Card description text
        
    Returns:
        HTML string for the capability card
    """
    return f"""
    <div class='intel-card'>
        <div style='margin-bottom:0.6rem;'>{icon_svg}</div>
        <div style='font-family: Playfair Display, serif; font-size: 1rem; font-weight: 600;
                    color: var(--text); margin-bottom: 0.5rem;'>{title}</div>
        <div class='card-body'>{body}</div>
    </div>
    """

def recommendation_card(icon_svg, title, body):
    """
    Create a recommendation card with icon, title, and description.
    
    Args:
        icon_svg: SVG icon HTML string
        title: Recommendation title
        body: Recommendation description text
        
    Returns:
        HTML string for the recommendation card
    """
    return f"""
    <div class='rec-card'>
        <div style='display:flex; align-items:center; gap:0.5rem; font-weight:700;
                    margin-bottom:0.4rem;'>{icon_svg} {title}</div>
        <span style='color: var(--text-dim);'>{body}</span>
    </div>
    """

def section_title(text):
    """
    Create a section title with accent border.
    
    Args:
        text: Section title text
        
    Returns:
        HTML string for the section title
    """
    return f"<div class='section-title'>{text}</div>"

def page_title(text):
    """
    Create a page title.
    
    Args:
        text: Page title text
        
    Returns:
        HTML string for the page title
    """
    return f"<div class='page-title'>{text}</div>"

def page_subtitle(text):
    """
    Create a page subtitle.
    
    Args:
        text: Page subtitle text
        
    Returns:
        HTML string for the page subtitle
    """
    return f"<div class='page-subtitle'>{text}</div>"
