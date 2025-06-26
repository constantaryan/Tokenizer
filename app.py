import os
import logging
import html
from flask import Flask, render_template, request, jsonify
from tokenizers import BasicTokenizer, RegexTokenizer, GPT4Tokenizer

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Initialize tokenizers
basic_tokenizer = BasicTokenizer()
regex_tokenizer = RegexTokenizer()
gpt4_tokenizer = GPT4Tokenizer()

# Sample training text for demonstration
SAMPLE_TEXT = """Hello world! This is a sample text for training our tokenizers. 
It contains various characters, numbers like 123, and symbols !@#$%^&*(). 
We use this to build our vocabulary for tokenization."""

# Train the tokenizers with sample text
basic_tokenizer.train(SAMPLE_TEXT, 300, verbose=False)
regex_tokenizer.train(SAMPLE_TEXT, 300, verbose=False)

def create_colored_tokens(text, tokens, tokenizer):
    """Create colored HTML representation of tokens with consistent colors per token ID"""
    colors = ['#4285f4', '#34a853', '#fbbc04', '#ea4335', '#9c27b0', '#ff9800', '#607d8b', '#795548', 
              '#3f51b5', '#009688', '#8bc34a', '#ffc107', '#ff5722', '#e91e63', '#673ab7', '#00bcd4']
    
    try:
        colored_html = ""
        
        # Create a mapping of unique token IDs to colors
        unique_tokens = list(set(tokens))
        token_color_map = {}
        for i, token_id in enumerate(unique_tokens):
            token_color_map[token_id] = colors[i % len(colors)]
        
        # Decode tokens back to text and apply consistent colors
        for token_id in tokens:
            try:
                # Get the color for this specific token ID
                color = token_color_map[token_id]
                
                # Special handling for GPT4 tokenizer
                if isinstance(tokenizer, GPT4Tokenizer):
                    try:
                        # Use the tokenizer's decode method for single token
                        token_text = tokenizer.decode([token_id])
                    except:
                        # Fallback if decode fails
                        token_text = f"[{token_id}]"
                else:
                    # For BasicTokenizer and RegexTokenizer
                    if hasattr(tokenizer, 'vocab') and token_id in tokenizer.vocab:
                        token_bytes = tokenizer.vocab[token_id]
                        token_text = token_bytes.decode('utf-8', errors='replace')
                    else:
                        # Fallback for tokens not in vocab
                        token_text = f"[{token_id}]"
                
                # Escape HTML and create colored span
                escaped_text = html.escape(token_text)
                colored_html += f'<span style="background-color: {color}; color: white; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Token ID: {token_id}">{escaped_text}</span>'
                
            except Exception as token_error:
                logging.warning(f"Error processing token {token_id}: {token_error}")
                # Fallback color for problematic tokens
                color = colors[0]
                colored_html += f'<span style="background-color: {color}; color: white; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Token ID: {token_id}">[{token_id}]</span>'
        
        return colored_html
        
    except Exception as e:
        logging.error(f"Error creating colored tokens: {e}")
        # Ultimate fallback: just return escaped text
        return html.escape(text)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tokenize', methods=['POST'])
def tokenize():
    # Get text from main textarea
    text = request.form.get('text', '').strip()
    tokenizer_type = request.form.get('tokenizer', 'basic')
    
    # Use the provided text directly
    combined_text = text
    
    # If no text provided, return empty state
    if not combined_text:
        return render_template('index.html', 
                             text='',
                             tokens=[],
                             tokenizer_type=tokenizer_type,
                             token_count=0,
                             colored_tokens='')
    
    try:
        if tokenizer_type == 'basic':
            tokens = basic_tokenizer.encode(combined_text)
            tokenizer_name = "Basic Tokenizer"
            tokenizer_obj = basic_tokenizer
        elif tokenizer_type == 'regex':
            tokens = regex_tokenizer.encode(combined_text)
            tokenizer_name = "Regex Tokenizer"
            tokenizer_obj = regex_tokenizer
        elif tokenizer_type == 'gpt4':
            tokens = gpt4_tokenizer.encode(combined_text)
            tokenizer_name = "GPT4 Tokenizer"
            tokenizer_obj = gpt4_tokenizer
        else:
            tokens = basic_tokenizer.encode(combined_text)
            tokenizer_name = "Basic Tokenizer"
            tokenizer_obj = basic_tokenizer
        
        # Generate colored token visualization
        colored_tokens = create_colored_tokens(combined_text, tokens, tokenizer_obj)
        
        return render_template('index.html', 
                             text=combined_text, 
                             tokens=tokens, 
                             tokenizer_type=tokenizer_type,
                             tokenizer_name=tokenizer_name,
                             token_count=len(tokens),
                             colored_tokens=colored_tokens)
    
    except Exception as e:
        logging.error(f"Error tokenizing text: {str(e)}")
        return render_template('index.html', 
                             text=combined_text,
                             tokenizer_type=tokenizer_type,
                             error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
