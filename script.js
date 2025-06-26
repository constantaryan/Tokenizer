// Auto-submit form when text changes
document.addEventListener('DOMContentLoaded', function() {
    const mainTextarea = document.getElementById('mainTextarea');
    const hiddenTokenizerInput = document.querySelector('input[name="tokenizer"]');
    const dropdownSelect = document.getElementById('tokenizerDropdown');
    const mainForm = document.getElementById('mainForm');
    
    // Update hidden input when dropdown changes and auto-submit
    if (dropdownSelect && hiddenTokenizerInput) {
        dropdownSelect.addEventListener('change', function() {
            hiddenTokenizerInput.value = this.value;
            // Auto-submit when tokenizer changes
            if (mainTextarea && mainTextarea.value.trim()) {
                mainForm.submit();
            }
        });
    }
    
    // Auto-submit on text input with debounce
    if (mainTextarea) {
        let timeoutId;
        mainTextarea.addEventListener('input', function() {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => {
                // Submit if there's content
                if (this.value.trim()) {
                    mainForm.submit();
                }
            }, 800); // 0.8 second delay for responsive feel
        });
        
        // Also submit when user stops typing (on blur)
        mainTextarea.addEventListener('blur', function() {
            if (this.value.trim()) {
                mainForm.submit();
            }
        });
    }
    
    // Sync tokenizer selection
    const initialTokenizer = hiddenTokenizerInput ? hiddenTokenizerInput.value : 'basic';
    if (dropdownSelect) {
        dropdownSelect.value = initialTokenizer;
    }
});

// Show/hide whitespace functionality
document.addEventListener('DOMContentLoaded', function() {
    const whitespaceToggle = document.getElementById('showWhitespace');
    const coloredText = document.querySelector('.colored-text');
    
    if (whitespaceToggle && coloredText) {
        whitespaceToggle.addEventListener('change', function() {
            if (this.checked) {
                // Show whitespace characters
                coloredText.style.whiteSpace = 'pre';
                coloredText.innerHTML = coloredText.innerHTML.replace(/ /g, '·').replace(/\n/g, '↵\n');
            } else {
                // Hide whitespace characters
                coloredText.style.whiteSpace = 'pre-wrap';
                coloredText.innerHTML = coloredText.innerHTML.replace(/·/g, ' ').replace(/↵/g, '');
            }
        });
    }
});
