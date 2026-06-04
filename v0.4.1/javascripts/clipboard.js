// clipboard.js: Used to solve the problem where copying notebook code blocks does not preserve the original formatting (such as line breaks and indentation).
/**
 * Copy text to clipboard with fallback support
 * @param {string} copyText - The text to copy to clipboard
 */
function copyToClipboard(copyText) {
    // Try modern clipboard API first
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard
            .writeText(copyText)
            .then(() => {
                console.log('Clipboard copy successful')
            })
            .catch((error) => {
                console.warn('Clipboard API failed, falling back to execCommand:', error)
                fallbackCopyToClipboard(copyText)
            })
    } else {
        // Fallback for older browsers or non-secure contexts
        fallbackCopyToClipboard(copyText)
    }
}

/**
 * Fallback method using execCommand for older browsers
 * @param {string} copyText - The text to copy to clipboard
 */
function fallbackCopyToClipboard(copyText) {
    const input = document.createElement('input')
    input.style.position = 'fixed'
    input.style.left = '-9999px'
    input.style.top = '-9999px'
    input.setAttribute('value', copyText)
    document.body.appendChild(input)
    
    try {
        input.select()
        input.setSelectionRange(0, 99999) // For mobile devices
        
        const successful = document.execCommand('copy')
        if (successful) {
            console.log('Fallback copy successful')
        } else {
            console.error('Fallback copy failed')
        }
    } catch (error) {
        console.error('Copy failed:', error)
    } finally {
        document.body.removeChild(input)
    }
}

// Listen for custom clipboard copy events
document.addEventListener('clipboard-copy', function (event) {
    const targetCopyId = event.target.getAttribute("for")
    if (!targetCopyId) {
        console.warn('No target ID found for clipboard copy')
        return
    }
    
    const targetDiv = document.querySelector(`#${targetCopyId}`)
    if (!targetDiv) {
        console.warn(`Target element with ID "${targetCopyId}" not found`)
        return
    }
    
    const previousElement = targetDiv.previousElementSibling
    if (!previousElement) {
        console.warn('No previous sibling element found to copy')
        return
    }
    
    const textToCopy = previousElement.innerText || previousElement.textContent
    if (!textToCopy) {
        console.warn('No text content found to copy')
        return
    }
    
    copyToClipboard(textToCopy)
})
