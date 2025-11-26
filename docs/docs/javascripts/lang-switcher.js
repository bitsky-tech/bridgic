window.addEventListener('DOMContentLoaded', function() {
  // Handles language URL rewrites for:
  // 1) Versioned docs (e.g. /v0.1.1/...).
  // 2) Special channels such as /latest/.
  // 3) Pages that already contain a language segment.
  // 4) Root-level pages without explicit versioning.
  const CONFIG = Object.freeze({
    languages: ['zh'],
    langSelector: 'a[hreflang="zh"]',
    versionPattern: /^v?\d+(\.\d+)*$/i,
    specialVersions: ['latest'],
  });

  const langAnchors = document.querySelectorAll(CONFIG.langSelector);
  const currentPath = window.location.pathname || '/';
  const hasTrailingSlash = currentPath.endsWith('/') && currentPath !== '/';
  const rawSegments = currentPath.split('/').filter(Boolean);
  let languageRemoved = false;
  const sanitizedSegments = rawSegments.filter(function(segment) {
    if (!languageRemoved && CONFIG.languages.includes(segment)) {
      languageRemoved = true;
      return false;
    }
    return true;
  });
  const insertIndex =
    sanitizedSegments.length &&
    (CONFIG.versionPattern.test(sanitizedSegments[0]) ||
      CONFIG.specialVersions.includes(sanitizedSegments[0].toLowerCase()))
      ? 1
      : 0;
  const searchAndHash = (window.location.search || '') + (window.location.hash || '');

  function buildLocalizedPath(lang) {
    const segments = sanitizedSegments.slice();
    segments.splice(insertIndex, 0, lang);
    const basePath = '/' + segments.join('/');
    const normalizedPath = segments.length ? basePath : '/';
    const finalPath = hasTrailingSlash && normalizedPath !== '/' ? normalizedPath + '/' : normalizedPath;
    return finalPath + searchAndHash;
  }

  langAnchors.forEach(function(anchor) {
    const lang = anchor.getAttribute('hreflang');
    if (!lang || !CONFIG.languages.includes(lang)) {
      return;
    }
    anchor.setAttribute('href', buildLocalizedPath(lang));
  });
});