"""
Tech Product Review Analyzer - UPDATED VERSION
==============================================
✅ Merge system cleaning logic (removes header/metadata rows)
✅ Enhanced About page explaining the system
✅ Same UI, system logic, and features
✅ Ready to deploy
"""

import os
import sys
import streamlit as st
from groq import Groq
import json
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import pandas as pd
import time
import re
from collections import Counter

st.set_page_config(
    page_title="Tech Review Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    body { font-size: 16px; }
    .enterprise-header {
        background: linear-gradient(135deg, #1E40AF 0%, #1E3A8A 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 8px;
        margin-bottom: 2rem;
    }
    .enterprise-header h1 { margin: 0; font-size: 2.5rem; font-weight: 700; }
    .metric-value { font-size: 2.8rem; font-weight: 700; color: #1E40AF; margin: 0.5rem 0; }
    .metric-label { color: #64748B; font-size: 0.95rem; font-weight: 600; text-transform: uppercase; }
    .stButton > button { background-color: #1E40AF; color: white; font-size: 1.05rem; padding: 0.9rem 1.8rem; font-weight: 600; }
    .stDataFrame { height: auto !important; }
</style>
""", unsafe_allow_html=True)

if 'step' not in st.session_state:
    st.session_state.step = 'input'

ENHANCED_CATEGORIES = {
    "Software & SaaS": {"keywords": ["app", "software", "subscription", "platform"], "confidence_boost": 1.5},
    "Smart Devices": {"keywords": ["smart", "iot", "connected"], "confidence_boost": 1.3},
    "Electronics": {"keywords": ["battery", "camera", "display", "phone"], "confidence_boost": 1.0},
    "Fitness Tech": {"keywords": ["fitness", "workout", "gym", "tracker"], "confidence_boost": 1.2},
    "Gaming": {"keywords": ["game", "console", "graphics"], "confidence_boost": 1.1},
    "Education Tech": {"keywords": ["learning", "tutor", "education"], "confidence_boost": 1.2}
}

# ============================================================================
# MERGE SYSTEM CLEANING LOGIC - HEADER/METADATA REMOVAL
# ============================================================================
COMMON_HEADER_PATTERNS = [
    r'PRODUCT REVIEWS DATASET', r'COLLECTION DATE:', r'TOTAL REVIEWS: \d+', r'TECH CATEGORY',
    r'REVIEWS DATASET', r'DATASET COLLECTION', r'FILE:.*\.(csv|txt|json)', r'REPORT GENERATED:',
    r'header:.*', r'footer:.*', r'confidential', r'proprietary', r'page.*\d+.*of.*\d+',
    r'report.*analysis', r'document.*id', r'date:.*\d{4}', r'timestamp:.*'
]

def clean_reviews_enhanced(reviews: List[str]) -> List[str]:
    """Enhanced review cleaning with header/metadata removal from merge system"""
    cleaned = []
    exclude_keywords = ['review', 'text', 'comment', 'feedback', 'review_text', 'customer', 'user', 'date']

    for review in reviews:
        if not review or not isinstance(review, str):
            continue

        clean_review = review.strip()

        # Skip if too short
        if len(clean_review) < 15:
            continue

        clean_lower = clean_review.lower()

        # CRITICAL: Skip header/metadata content using common patterns
        if any(re.search(pattern, clean_lower) for pattern in COMMON_HEADER_PATTERNS):
            continue

        # Skip if matches exclude patterns and too short
        if any(kw in clean_lower for kw in exclude_keywords) and len(clean_review) < 50:
            continue

        # Skip if looks like header (all caps short text)
        if clean_review.isupper() and len(clean_review) < 100:
            continue

        cleaned.append(clean_review)

    # Remove duplicates while preserving order
    seen = set()
    unique_cleaned = []
    for review in cleaned:
        if review not in seen:
            seen.add(review)
            unique_cleaned.append(review)

    return unique_cleaned

def clean_json_reviews_aggressive(reviews: List[str]) -> List[str]:
    """Aggressive cleaning for JSON files"""
    cleaned = []
    rating_patterns = [r'⭐', r'★', r'[0-9]/[0-9]', r'rating:', r'score:']

    for review in reviews:
        if not review or not isinstance(review, str):
            continue

        clean_review = review.strip()
        if len(clean_review) < 20:
            continue

        clean_lower = clean_review.lower()

        # Skip header patterns
        if any(re.search(pattern, clean_lower) for pattern in COMMON_HEADER_PATTERNS):
            continue

        # Skip rating patterns
        if any(re.search(pattern, clean_lower) for pattern in rating_patterns):
            continue

        # Skip technical IDs
        if re.search(r'[a-z]+_[a-z0-9]+', clean_lower) and len(clean_review) < 50:
            continue

        # Skip all-caps short text
        if clean_review.isupper() and len(clean_review) < 100:
            continue

        # Skip mostly symbols
        symbol_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', clean_review)) / len(clean_review)
        if symbol_ratio > 0.5 and len(clean_review) < 80:
            continue

        cleaned.append(clean_review)

    # Remove duplicates
    seen = set()
    unique_cleaned = []
    for review in cleaned:
        if review not in seen:
            seen.add(review)
            unique_cleaned.append(review)

    return unique_cleaned

# ============================================================================
# EXISTING SYSTEM FUNCTIONS
# ============================================================================

import os
st.write("GROQ_API_KEY is", GROQ_API_KEY )

import os
api_key = os.environ.get("GROQ_API_KEY")
st.write("GROQ_API_KEY is", api_key)

@st.cache_resource
def get_groq_client():
    return Groq(api_key=api_key)

client = get_groq_client()

def detect_category(review_text: str) -> Tuple[str, int]:
    review_lower = review_text.lower()
    scores = {}
    for category, data in ENHANCED_CATEGORIES.items():
        keywords = data["keywords"]
        boost = data.get("confidence_boost", 1.0)
        positive_matches = sum(1 for kw in keywords if kw in review_lower)
        score = positive_matches * boost
        scores[category] = max(0, score)
    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]
    confidence = min(100, int(best_score * 20)) if best_score > 0 else 40
    return best_category, confidence

def analyze_review(review_text: str, category: str) -> Dict:
    prompt = f"""Analyze this tech review. Return ONLY valid JSON:
{{
    "sentiment": "Positive|Negative|Mixed",
    "confidence": 0.85,
    "summary": "One sentence",
    "strengths": ["strength"],
    "issues": ["issue"],
    "recommendation": "Recommendation",
    "recommendation_priority": "High|Medium|Low"
}}
CATEGORY: {category}
REVIEW: "{review_text[:300]}"
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return result
    except:
        return None

def extract_reviews_from_list(data_list: List) -> List[str]:
    reviews = []
    for item in data_list:
        if isinstance(item, dict):
            for key in ['text', 'review', 'comment', 'feedback', 'content', 'review_text', 'description']:
                if key in item and item[key] and isinstance(item[key], str):
                    reviews.append(str(item[key]))
                    break
        elif isinstance(item, str) and len(item) > 20:
            reviews.append(item)
    return reviews

def extract_reviews_from_dict(data: Dict, max_depth: int = 3) -> List[str]:
    reviews = []
    if max_depth <= 0:
        return reviews
    for key, value in data.items():
        if isinstance(value, str) and len(value) > 25:
            if key.lower() not in ['id', 'date', 'timestamp', 'version', 'metadata']:
                reviews.append(value)
        elif isinstance(value, dict):
            reviews.extend(extract_reviews_from_dict(value, max_depth - 1))
        elif isinstance(value, list):
            reviews.extend(extract_reviews_from_list(value))
    return reviews

def process_file(uploaded_file) -> Tuple[Optional[List[str]], Optional[Dict]]:
    file_type = uploaded_file.name.split('.')[-1].lower()
    metadata = {"filename": uploaded_file.name, "file_type": file_type, "original_rows": 0, "cleaned_rows": 0}

    try:
        if file_type in ['csv', 'xlsx']:
            if file_type == 'csv':
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            metadata["original_rows"] = len(df)
            text_columns = df.select_dtypes(include=['object']).columns
            review_cols = [col for col in text_columns if any(kw in col.lower() for kw in ['review', 'comment', 'text'])]
            if not review_cols:
                return None, metadata
            reviews = df[review_cols[0]].dropna().astype(str).tolist()
            # APPLY CLEANING
            reviews = clean_reviews_enhanced(reviews)
            metadata["cleaned_rows"] = len(reviews)
            return reviews, metadata

        elif file_type == 'txt':
            content = uploaded_file.read().decode('utf-8')
            reviews = [p.strip() for p in re.split(r'\n\s*\n', content) if len(p.strip()) >= 20]
            if not reviews:
                reviews = [line.strip() for line in content.split('\n') if len(line.strip()) >= 20]
            # APPLY CLEANING
            reviews = clean_reviews_enhanced(reviews)
            metadata["cleaned_rows"] = len(reviews)
            metadata["original_rows"] = len(reviews)
            return reviews, metadata

        elif file_type == 'json':
            content = uploaded_file.read().decode('utf-8')
            data = json.loads(content)
            reviews = []
            if isinstance(data, list):
                reviews = extract_reviews_from_list(data)
            elif isinstance(data, dict):
                if 'reviews' in data and isinstance(data['reviews'], list):
                    reviews = extract_reviews_from_list(data['reviews'])
                elif 'feedback' in data and isinstance(data['feedback'], list):
                    reviews = extract_reviews_from_list(data['feedback'])
                elif 'comments' in data and isinstance(data['comments'], list):
                    reviews = extract_reviews_from_list(data['comments'])
                else:
                    reviews = extract_reviews_from_dict(data)

            metadata["original_rows"] = len(reviews)
            # APPLY AGGRESSIVE JSON CLEANING
            reviews = clean_json_reviews_aggressive(reviews)
            metadata["cleaned_rows"] = len(reviews)
            return reviews, metadata

    except:
        pass

    return None, metadata

def generate_simple_report(results: List[Dict], sentiments: Counter, categories: Counter) -> str:
    total = len(results)
    pos = sentiments.get("Positive", 0)
    neg = sentiments.get("Negative", 0)
    mixed = sentiments.get("Mixed", 0)
    avg_conf = sum(r.get("confidence", 0) for r in results) / total if total > 0 else 0

    issues = [r.get("issues", []) for r in results if r.get("issues")]
    all_issues = [item for sublist in issues for item in sublist if item != "N/A"]
    issue_counter = Counter(all_issues)

    strengths = [r.get("strengths", []) for r in results if r.get("strengths")]
    all_strengths = [item for sublist in strengths for item in sublist if item != "N/A"]
    strength_counter = Counter(all_strengths)

    report = f"REVIEW ANALYSIS REPORT\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nTotal Reviews: {total}\n\nSUMMARY\nPositive: {pos}\nNegative: {neg}\nMixed: {mixed}\nAvg Confidence: {avg_conf:.0%}\n\nCATEGORIES\n"

    for cat, cnt in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        percentage = (cnt / total) * 100
        report += f"{cat}: {cnt} ({percentage:.0f}%)\n"

    report += "\nTOP ISSUES\n"
    if issue_counter:
        for idx, (issue, count) in enumerate(issue_counter.most_common(10), 1):
            percentage = (count / total) * 100
            report += f"{idx}. {issue} - {count} ({percentage:.0f}%)\n"

    report += "\nTOP STRENGTHS\n"
    if strength_counter:
        for idx, (strength, count) in enumerate(strength_counter.most_common(10), 1):
            percentage = (count / total) * 100
            report += f"{idx}. {strength} - {count} ({percentage:.0f}%)\n"

    return report

def render_header():
    st.markdown("""<div class="enterprise-header">
    <h1>Tech Product Review Analyzer</h1>
    <p>Simple • Fast • Clean Results</p>
    </div>""", unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("### NAVIGATION")
        page = st.radio("Select:", ["Single Review", "Batch Processing", "About"], label_visibility="collapsed")
        st.divider()
        st.markdown("### TARGET AUDIENCE")
        st.caption("Engineers at tech companies")
        st.caption("Product Managers")
        st.caption("Business Analysts")
        return page

def render_about_page():
    st.markdown("## About This System")

    st.markdown("""
    ### 🎯 What This System Does
    
    **Tech Product Review Analyzer** helps businesses and teams quickly understand customer feedback 
    from tech product reviews. It automatically analyzes reviews to identify what customers love, 
    what problems they're having, and what improvements they want.
    """)

    st.markdown("""
    ### 🔄 How It Works
    
    1. **Single Review Analysis**: Paste any customer review to get instant insights
    2. **Batch Processing**: Upload files with multiple reviews (CSV, Excel, JSON, TXT)
    3. **Smart Cleaning**: Automatically removes headers, metadata, and non-review content
    4. **AI Analysis**: Detects sentiment, finds key issues and strengths
    5. **Export Results**: Download clean data for your team
    """)

    st.markdown("### ✨ Key Features")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📁 Data Processing**")
        st.markdown("- CSV, Excel, JSON, TXT support")
        st.markdown("- Auto-detect review columns")
        st.markdown("- Remove headers/metadata")
        st.markdown("- Filter duplicate reviews")
        
        st.markdown("**🎯 Analysis**")
        st.markdown("- Sentiment detection")
        st.markdown("- Issue identification")
        st.markdown("- Strength extraction")
        st.markdown("- Product categorization")

    with col2:
        st.markdown("**📊 Results**")
        st.markdown("- Clean CSV exports")
        st.markdown("- Summary text reports")
        st.markdown("- Top issues & strengths")
        st.markdown("- Category breakdowns")
        
        st.markdown("**⚡ Performance**")
        st.markdown("- Fast AI processing")
        st.markdown("- Real-time progress")
        st.markdown("- Professional interface")
        st.markdown("- Easy to use")

    st.markdown("""
    ### 📊 Data Quality
    
    The system automatically cleans your data by:
    - Removing dataset headers and metadata rows
    - Filtering out duplicate reviews
    - Skipping empty or too-short entries
    - Keeping only genuine customer feedback
    - Categorizing products automatically
    """)

    st.markdown("""
    ### 💾 Export Options
    
    - **CSV Results**: Clean data export without headers (ready for analysis)
    - **Text Reports**: Summary of key findings with top issues and strengths
    - **Simple Formats**: Easy-to-use files that work with any tool
    """)

    st.markdown("### 🚀 How to Use")

    with st.expander("📝 Single Review Analysis"):
        st.markdown("""
        1. Go to **Single Review** tab
        2. Paste a customer review in the text box
        3. Click **Analyze** button
        4. View sentiment, issues, and strengths instantly
        """)

    with st.expander("📁 Batch File Processing"):
        st.markdown("""
        1. Go to **Batch Processing** tab  
        2. Upload your file (CSV, Excel, JSON, or TXT)
        3. System automatically detects review columns
        4. Set how many reviews to analyze
        5. Click **Analyze** 
        6. Download CSV data or text report
        """)

    st.markdown("""
    ### 🎪 Perfect For
    
    - **Product Teams**: Understand what features customers want
    - **Support Teams**: Identify common customer problems  
    - **Business Analysts**: Spot trends in customer feedback
    - **Quality Assurance**: Find product weaknesses
    - **Marketing Teams**: Discover what customers love
    """)

    st.markdown("""
    ### ⚙️ Behind the Scenes
    
    - **AI Engine**: Groq API with advanced language model
    - **Data Processing**: Smart cleaning and filtering
    - **File Support**: CSV, Excel, JSON, and text files
    - **Security**: Your data stays private and secure
    """)

    st.markdown("""
    ### ✅ What Makes It Different
    
    - **No Complex Setup**: Just upload and analyze
    - **Automatic Cleaning**: No manual data cleaning needed
    - **Practical Results**: Focus on actionable insights
    - **Clean Exports**: Ready-to-use data without headers
    - **User-Friendly**: Works for both technical and non-technical users
    """)

def render_batch_with_report():
    st.markdown("## Batch File Processing")

    uploaded_file = st.file_uploader("Choose file:", type=["csv", "xlsx", "json", "txt"])

    if uploaded_file:
        with st.spinner("Processing file..."):
            reviews, metadata = process_file(uploaded_file)

            if reviews and metadata:
                st.markdown("### File Analysis")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f'<div class="metric-label">Original</div><div class="metric-value">{metadata["original_rows"]}</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown(f'<div class="metric-label">Cleaned</div><div class="metric-value">{len(reviews)}</div>', unsafe_allow_html=True)

                with col3:
                    if metadata["original_rows"] == 0:
                        quality = 0
                    else:
                        quality = (len(reviews) / metadata["original_rows"]) * 100
                    st.markdown(f'<div class="metric-label">Quality %</div><div class="metric-value">{min(quality, 100):.0f}%</div>', unsafe_allow_html=True)

                with col4:
                    st.markdown(f'<div class="metric-label">Type</div><div class="metric-value">{metadata["file_type"].upper()}</div>', unsafe_allow_html=True)

                st.divider()

                col1, col2 = st.columns([2, 1])
                with col1:
                    limit = st.number_input("Reviews to analyze:", 1, len(reviews), min(20, len(reviews)))
                with col2:
                    start = st.button("Analyze", use_container_width=True)

                if start:
                    progress = st.progress(0)
                    status = st.empty()
                    results = []

                    for idx, review in enumerate(reviews[:limit]):
                        progress.progress((idx + 1) / limit)
                        status.text(f"Analyzing {idx + 1}/{limit}...")

                        category, _ = detect_category(review)
                        result = analyze_review(review, category)

                        if result:
                            result["category"] = category
                            result["review_text"] = review[:100]
                            results.append(result)

                        time.sleep(0.1)

                    progress.progress(1.0)
                    status.text("Complete!")
                    st.divider()

                    st.markdown("### Results")

                    sentiments = Counter([r.get("sentiment", "Unknown") for r in results])
                    categories = Counter([r.get("category", "Unknown") for r in results])

                    col_left, col_right = st.columns(2)

                    with col_left:
                        st.markdown("**Results**")
                        pos = sentiments.get("Positive", 0)
                        neg = sentiments.get("Negative", 0)
                        mixed = sentiments.get("Mixed", 0)
                        st.markdown(f"- Positive: {pos}")
                        st.markdown(f"- Negative: {neg}")
                        st.markdown(f"- Mixed: {mixed}")

                    with col_right:
                        st.markdown("**Categories**")
                        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                            percentage = (count / len(results)) * 100
                            st.markdown(f"- {category}: {count} ({percentage:.0f}%)")

                    st.divider()
                    st.markdown("### Issues & Strengths")

                    issues = [r.get("issues", []) for r in results if r.get("issues")]
                    all_issues = [item for sublist in issues for item in sublist if item != "N/A"]
                    issue_counter = Counter(all_issues)

                    strengths = [r.get("strengths", []) for r in results if r.get("strengths")]
                    all_strengths = [item for sublist in strengths for item in sublist if item != "N/A"]
                    strength_counter = Counter(all_strengths)

                    col_left, col_right = st.columns(2)

                    with col_left:
                        st.markdown("**Top Issues**")
                        if issue_counter:
                            for idx, (issue, count) in enumerate(issue_counter.most_common(5), 1):
                                st.markdown(f"{idx}. {issue} - {count}")

                    with col_right:
                        st.markdown("**Top Strengths**")
                        if strength_counter:
                            for idx, (strength, count) in enumerate(strength_counter.most_common(5), 1):
                                st.markdown(f"{idx}. {strength} - {count}")

                    st.divider()

                    st.markdown("### All Reviews")
                    df_display = pd.DataFrame([{
                        'Review': r.get('review_text', 'N/A'),
                        'Category': r.get('category', 'Unknown'),
                        'Sentiment': r.get('sentiment', 'Unknown'),
                        'Confidence': f"{r.get('confidence', 0):.0%}",
                        'Issues': ', '.join(r.get('issues', [])),
                        'Strengths': ', '.join(r.get('strengths', []))
                    } for r in results])

                    st.dataframe(df_display, use_container_width=True, height=None)

                    st.divider()

                    st.markdown("### Download Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        # Merge system CSV logic - NO header row
                        csv_with_headers = df_display.to_csv(index=False)
                        csv_lines = csv_with_headers.split('\n')
                        csv_no_headers = '\n'.join(csv_lines[1:])
                        st.download_button("📊 CSV Results", csv_no_headers, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", use_container_width=True)

                    with col2:
                        report = generate_simple_report(results, sentiments, categories)
                        st.download_button("📄 Text Report", report, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", use_container_width=True)

def render_single_with_report():
    st.markdown("## Single Review Analysis")
    review_text = st.text_area("Paste Review:", height=150, placeholder="Paste review here...")

    if st.button("Analyze", use_container_width=True) and review_text:
        with st.spinner("Analyzing..."):
            category, conf = detect_category(review_text)
            result = analyze_review(review_text, category)

            if result:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f'<div class="metric-label">Sentiment</div><div class="metric-value">{result.get("sentiment")}</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown(f'<div class="metric-label">Confidence</div><div class="metric-value">{result.get("confidence", 0):.0%}</div>', unsafe_allow_html=True)

                with col3:
                    st.markdown(f'<div class="metric-label">Category</div><div class="metric-value">{category}</div>', unsafe_allow_html=True)

                with col4:
                    st.markdown(f'<div class="metric-label">Priority</div><div class="metric-value">{result.get("recommendation_priority")}</div>', unsafe_allow_html=True)

                st.divider()

                st.markdown("### Summary")
                st.info(result.get("summary", "N/A"))

                st.markdown("### Recommendation")
                st.success(result.get("recommendation", "N/A"))

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Strengths")
                    for s in result.get("strengths", []):
                        st.markdown(f"- {s}")

                with col2:
                    st.markdown("### Issues")
                    for i in result.get("issues", []):
                        st.markdown(f"- {i}")

def main():
    render_header()
    page = render_sidebar()

    if page == "Single Review":
        render_single_with_report()
    elif page == "Batch Processing":
        render_batch_with_report()
    elif page == "About":
        render_about_page()

if __name__ == "__main__":
    main()
