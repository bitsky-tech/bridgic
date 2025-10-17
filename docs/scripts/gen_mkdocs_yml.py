"""
Safe API documentation generation script
Uses staging and recovery mechanism to protect other configurations
"""

import logging
import sys
import yaml
import os
import shutil
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any
import re

import mkdocs_gen_files
import ast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

LLM_OVERVIEW_PATH = "extras/llms/index.md"

class DocumentationConfig:
    """Documentation generation configuration class"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Set default configuration
        self._set_defaults()
        
        # Load configuration file if provided
        config_file = Path(config_file) if config_file else Path(__file__).parent / "doc_config.yaml"

        if config_file.exists():
            self.load_from_file(str(config_file))
        else:
            logger.error(f"Configuration file does not exist: {config_file}")
            raise FileNotFoundError(f"Configuration file does not exist: {config_file}")
    
    def _set_defaults(self):
        """Set default configuration values"""
        # Excluded file name patterns
        self.exclude_patterns = {
            "__pycache__",
            ".venv", 
            "venv",
            ".git",
            ".pytest_cache",
            "node_modules",
            "tests",
            "test",
            "ipynb",
            "dist",
            "build"
        }
        
        # Excluded file names
        self.exclude_files = {
            "__main__.py",
            "setup.py",
            "conftest.py",
            "version.py"
        }
        
        # Package directories to process
        self.packages = ["bridgic-core"]
        self.package_info = {}
        
        # Base path for documentation generation
        self.docs_base_path = "reference"
        
        # Generation options
        self.verbose = True
        self.skip_empty_modules = True
        # mkdocstrings rendering options should be governed by mkdocs_template.yml only
        self.show_source = True
        self.show_root_heading = True
        self.show_root_toc_entry = True
        self.generate_index_pages = True
        # New option: generate only index pages for packages (directories with __init__.py)
        self.only_index_pages = True
        # New option: render single-entry groups as collapsible with an Index child for visibility
        self.single_entry_as_group = True
        self.docstring_style = "numpy"
        
        # mkdocstrings options centralized in mkdocs_template.yml; keep minimal defaults here if needed
        self.mkdocstrings_options = {}
    
    def load_from_file(self, config_file: str):
        """Load configuration from YAML configuration file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Update exclusion patterns
            if 'exclude_patterns' in config_data:
                self.exclude_patterns = set(config_data['exclude_patterns'])
                
            if 'exclude_files' in config_data:
                self.exclude_files = set(config_data['exclude_files'])
            
            # Update package configuration
            if 'packages' in config_data:
                packages_config = config_data['packages']
                if isinstance(packages_config, list):
                    if packages_config and isinstance(packages_config[0], dict):
                        # New format: list of dictionaries with detailed information
                        self.packages = [pkg['path'] for pkg in packages_config]
                        self.package_info = {
                            pkg['path']: {
                                'name': pkg.get('name', pkg['path']),
                                'description': pkg.get('description', '')
                            }
                            for pkg in packages_config
                        }
                    else:
                        # Old format: simple path list
                        self.packages = packages_config
            
            # Update generation options
            if 'generation_options' in config_data:
                gen_opts = config_data['generation_options']
                self.docs_base_path = gen_opts.get('docs_base_path', self.docs_base_path)
                self.verbose = gen_opts.get('verbose', self.verbose)
                self.skip_empty_modules = gen_opts.get('skip_empty_modules', self.skip_empty_modules)
                self.show_source = gen_opts.get('show_source', self.show_source)
                self.show_root_heading = gen_opts.get('show_root_heading', self.show_root_heading)
                self.show_root_toc_entry = gen_opts.get('show_root_toc_entry', self.show_root_toc_entry)
                self.generate_index_pages = gen_opts.get('generate_index_pages', self.generate_index_pages)
                self.docstring_style = gen_opts.get('docstring_style', self.docstring_style)
                self.only_index_pages = gen_opts.get('only_index_pages', self.only_index_pages)
                self.single_entry_as_group = gen_opts.get('single_entry_as_group', self.single_entry_as_group)
            
            # mkdocstrings_options now governed by mkdocs_template.yml; ignore here
                    
            logger.info(f"Loaded settings from configuration file: {config_file}")
                    
        except Exception as e:
            logger.error(f"Failed to load configuration file {config_file}: {e}")
            logger.info("Will use default configuration")
    
    def get_package_display_name(self, package_path: str) -> str:
        """Get the display name of the package"""
        if package_path in self.package_info:
            return self.package_info[package_path]['name']
        return Path(package_path).name.replace("-", "_")
    
    def get_package_description(self, package_path: str) -> str:
        """Get the description of the package"""
        if package_path in self.package_info:
            return self.package_info[package_path]['description']
        return ""

class SafeMkDocsConfigUpdater:
    """Safe MkDocs configuration file updater - uses staging and recovery mechanism"""
    
    def __init__(self, mkdocs_path: Path):
        self.mkdocs_path = mkdocs_path
    
    def update_mkdocs_config(self, nav_structure: Dict[str, Any]) -> bool:
        """Generate mkdocs.yml from template with API Reference content"""
        try:
            template_path = self.mkdocs_path.parent / "scripts" / "mkdocs_template.yml"
            
            if not template_path.exists():
                logger.error(f"Template file does not exist: {template_path}")
                return False
            
            # Read template file content
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            logger.debug("Starting template-based MkDocs configuration generation...")
            
            # Check if template placeholder exists
            if "{{API_REFERENCE_CONTENT}}" not in template_content:
                logger.error("Template placeholder {{API_REFERENCE_CONTENT}} not found in template")
                return False
            
            # Generate API Reference content
            api_reference_content = self._generate_api_reference_content(nav_structure)
            
            # Replace template placeholder with generated content
            final_content = template_content.replace("{{API_REFERENCE_CONTENT}}", api_reference_content)
            
            # Write to mkdocs.yml
            with open(self.mkdocs_path, 'w', encoding='utf-8') as f:
                f.write(final_content)
            
            logger.info("Successfully generated MkDocs configuration file from template")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate MkDocs configuration file from template: {e}")
            import traceback
            logger.debug(f"Detailed error information: {traceback.format_exc()}")
            return False
    
    def _update_mkdocs_config_legacy(self, nav_structure: Dict[str, Any]) -> bool:
        """Legacy method for updating MkDocs configuration (fallback)"""
        try:
            # Read original file content
            with open(self.mkdocs_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            logger.debug("Starting legacy MkDocs configuration update...")
            
            # 1. Staging: Use regex to separate nav section from other configurations
            # Find the position of nav:
            nav_start_pattern = r'^nav:\s*$'
            nav_match = re.search(nav_start_pattern, original_content, flags=re.MULTILINE)
            
            if not nav_match:
                logger.error("Cannot find nav configuration section")
                return False
            
            nav_start_pos = nav_match.end()
            
            # Separate content before nav
            content_before_nav = original_content[:nav_match.start()]
            
            # Find the first top-level configuration item after nav
            remaining_content = original_content[nav_start_pos:]
            next_config_pattern = r'^\n([a-z_]+:)'
            next_config_match = re.search(next_config_pattern, remaining_content, flags=re.MULTILINE)
            
            if next_config_match:
                nav_content = remaining_content[:next_config_match.start() + 1]  # keep the newline
                content_after_nav = remaining_content[next_config_match.start() + 1:]
            else:
                # nav is the last configuration item
                nav_content = remaining_content
                content_after_nav = ""
            
            logger.debug(f"Configuration separated: {len(content_before_nav)} chars before nav, {len(nav_content)} chars in nav content, {len(content_after_nav)} chars after nav")
            
            # 2. Process nav content: remove existing API Reference, keep other navigation items
            new_nav_content = self._rebuild_nav_content(nav_content, nav_structure)
            
            # 3. Reassemble complete content
            final_content = content_before_nav + "nav:\n" + new_nav_content
            if content_after_nav:
                final_content += content_after_nav
            
            # 4. Write back to file
            with open(self.mkdocs_path, 'w', encoding='utf-8') as f:
                f.write(final_content)
            
            logger.info("Successfully updated MkDocs configuration file using legacy method")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update MkDocs configuration file using legacy method: {e}")
            import traceback
            logger.debug(f"Detailed error information: {traceback.format_exc()}")
            return False
    
    def _rebuild_nav_content(self, nav_content: str, nav_structure: Dict[str, Any]) -> str:
        """Rebuild nav content, preserve non-API Reference parts, replace API Reference"""
        nav_lines = nav_content.split('\n')
        new_nav_lines = []
        skip_api_reference = False
        api_reference_found = False
        
        for line in nav_lines:
            stripped = line.strip()
            
            # Check if this is an API Reference line
            if stripped.startswith('- API Reference:'):
                skip_api_reference = True
                api_reference_found = True
                continue
            elif skip_api_reference:
                # Check if we've reached the next top-level navigation item (2-space indented "- ")
                if line.startswith('  - ') and not line.startswith('    '):
                    skip_api_reference = False
                    new_nav_lines.append(line)
                # Otherwise skip API Reference sub-items
            else:
                new_nav_lines.append(line)
        
        # Build new API Reference
        api_reference_nav = self._build_api_reference_nav(nav_structure)
        yaml_str = self._nav_to_yaml_string(api_reference_nav)
        new_api_section = f"  - API Reference:\n{yaml_str}"
        
        # Find About position to insert API Reference
        about_index = None
        for i, line in enumerate(new_nav_lines):
            if line.strip().startswith('- About:'):
                about_index = i
                break
        
        if about_index is not None:
            new_nav_lines.insert(about_index, new_api_section)
            new_nav_lines.insert(about_index + 1, '')  # Add empty line separator
        else:
            # If no About section, add at the end
            if new_nav_lines and new_nav_lines[-1].strip():
                new_nav_lines.append('')
            new_nav_lines.append(new_api_section)
        
        return '\n'.join(new_nav_lines)
    
    def _build_api_reference_nav(self, nav_structure: Dict[str, Any]) -> List[Any]:
        """Build navigation structure for API Reference"""
        api_nav = []
        
        for package_name, package_structure in nav_structure.items():
            formatted_name = self._format_display_name(package_name)
            package_nav = self._build_nav_structure(package_structure)
            api_nav.append({formatted_name: package_nav})
        
        return api_nav
    
    def _build_nav_structure(self, structure: Dict[str, Any]) -> List[Any]:
        """Recursively build navigation structure"""
        nav_items = []
        
        for key, value in sorted(structure.items()):
            if isinstance(value, dict):
                sub_nav = self._build_nav_structure(value)
                # Keep dotted python package path as-is, otherwise format
                formatted_name = key if ('.' in key) else self._format_display_name(key)
                nav_items.append({formatted_name: sub_nav})
            else:
                formatted_name = key if ('.' in key) else self._format_display_name(key)
                nav_items.append({formatted_name: value})
        
        return nav_items
    
    def _format_display_name(self, name: str) -> str:
        """Format display name"""
        formatted = name.replace('_', ' ').title()
        
        # Special handling
        replacements = {
            'Llm': 'LLM', 'Api': 'API', 'Ai': 'AI', 'Http': 'HTTP',
            'Json': 'JSON', 'Xml': 'XML', 'Url': 'URL', 'Uri': 'URI',
            'Uuid': 'UUID', 'Id': 'ID'
        }
        
        for old, new in replacements.items():
            formatted = formatted.replace(old, new)
            
        return formatted
    
    def _nav_to_yaml_string(self, nav_list: List[Any], indent: int = 2) -> str:
        """Convert navigation list to YAML string"""
        lines = []
        base_indent = " " * indent
        
        for item in nav_list:
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, list):
                        lines.append(f"{base_indent}- {key}:")
                        sub_yaml = self._nav_to_yaml_string(value, indent + 2)
                        if sub_yaml:
                            # Split sub_yaml into lines and add them
                            sub_lines = sub_yaml.split('\n')
                            for sub_line in sub_lines:
                                if sub_line.strip():  # Skip empty lines
                                    lines.append(sub_line)
                    else:
                        lines.append(f"{base_indent}- {key}: {value}")
            else:
                lines.append(f"{base_indent}- {item}")
        
        return "\n".join(lines)
    
    def _generate_api_reference_content(self, nav_structure: Dict[str, Any]) -> str:
        """Generate API Reference content from navigation structure"""
        try:
            # Build custom layout:
            # - Bridgic-Core: keep existing structure
            # - Bridgic-Integration:
            #     - llms:
            #         - bridgic.llms.openai: <path>
            #         - bridgic.llms.openai_like: <path>
            #         - bridgic.llms.vllm: <path>

            lines: List[str] = []

            # 1) Bridgic-Core (keep as-is if present)
            core_key_candidates = [k for k in nav_structure.keys() if k.lower() == 'bridgic-core']
            if core_key_candidates:
                core_key = core_key_candidates[0]
                core_nav = self._build_api_reference_nav({core_key: nav_structure[core_key]})
                for item in core_nav:
                    for key, value in item.items():
                        # 4 spaces under "- API Reference:"
                        lines.append(f"    - {self._format_display_name(key)}:")
                        if isinstance(value, list):
                            # Children start at 6 spaces
                            sub_yaml = self._nav_to_yaml_string(value, indent=6)
                            if sub_yaml:
                                for sub_line in sub_yaml.split('\n'):
                                    if sub_line.strip():
                                        lines.append(sub_line)

            # 2) Bridgic-Integration > llms
            # Derive entries from known integration packages present in nav_structure
            integration_entries: List[Tuple[str, str]] = []
            # Map of package name prefix to dotted module suffix
            # We look for packages that start with 'bridgic-llms-'
            for pkg_name in nav_structure.keys():
                if pkg_name.startswith('bridgic-llms-'):
                    # suffix like 'openai', 'openai-like', 'vllm' -> dotted module uses underscore
                    suffix = pkg_name.replace('bridgic-llms-', '')
                    dotted_suffix = suffix.replace('-', '_')
                    dotted = f"bridgic.llms.{dotted_suffix}"
                    # Build expected index path
                    index_path = f"reference/{pkg_name}/bridgic/llms/{dotted_suffix}/index.md"
                    integration_entries.append((dotted, index_path))

            if integration_entries:
                # 4 spaces for first level under API Reference
                lines.append("    - Bridgic-Integration:")
                # 6 spaces for second level
                lines.append("      - llms:")
                # Insert overview as first entry to avoid promotion of first LLM package
                lines.append(f"        - llms: {LLM_OVERVIEW_PATH}")
                # 8 spaces for entries
                for dotted, path in integration_entries:
                    lines.append(f"        - {dotted}: {path}")

            return '\n'.join(lines)
            
        except Exception as e:
            logger.error(f"Failed to generate API Reference content: {e}")
            return ""

class DocumentationGenerator:
    """Main documentation generator class"""
    
    def __init__(self, config: DocumentationConfig):
        self.config = config
        self.nav = mkdocs_gen_files.Nav()
        # script is located under docs/scripts/, so the root directory is the upper layer of the script directory
        self.root = Path(__file__).parent.parent.parent
        # docs directory is the upper layer of the script directory
        self.docs_dir = Path(__file__).parent.parent
        self.generated_files = []
        self.skipped_files = []
        self.error_files = []
        self.nav_structure = {}
        # Collect package directories (with __init__.py) per top-level package display name
        self.package_nodes: Dict[str, Dict[Tuple[str, ...], str]] = {}
        
    def clean_reference_directory(self) -> None:
        """Clean the reference directory before generating new documentation"""
        try:
            reference_dir = self.docs_dir / "docs" / self.config.docs_base_path
            
            if reference_dir.exists():
                logger.info(f"Cleaning reference directory: {reference_dir}")
                shutil.rmtree(reference_dir)
                logger.info("Successfully cleaned reference directory")
            else:
                logger.info(f"Reference directory does not exist, no cleanup needed: {reference_dir}")
                
        except PermissionError as e:
            logger.error(f"Permission denied while cleaning reference directory: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to clean reference directory: {e}")
            raise

    def should_exclude_path(self, path: Path) -> bool:
        """Check if path should be excluded"""
        for pattern in self.config.exclude_patterns:
            if pattern in path.parts:
                return True
                
        if path.name in self.config.exclude_files:
            return True
            
        if path.name.startswith('.'):
            return True
            
        return False
    
    def is_valid_python_module(self, path: Path) -> bool:
        """Check if it's a valid Python module"""
        if not path.suffix == '.py':
            return False
            
        try:
            if not path.exists() or path.stat().st_size == 0:
                if self.config.skip_empty_modules:
                    return False
        except (OSError, IOError):
            return False
            
        return True
    
    def generate_module_identifier(self, parts: List[str]) -> str:
        """Generate module identifier"""
        return ".".join(parts)
    
    def process_init_module(self, parts: List[str], doc_path: Path, full_doc_path: Path, source_path: Path) -> Tuple[List[str], Path, Path, bool]:
        """Process __init__.py module and check if it has __all__ defined"""
        has_all = False
        
        # Check if __init__.py has __all__ defined
        try:
            _, exports = self._parse_init_doc_and_all(source_path)
            has_all = len(exports) > 0
            if self.config.verbose and has_all:
                logger.debug(f"Found __all__ in {source_path}: {exports}")
        except Exception as e:
            if self.config.verbose:
                logger.debug(f"Failed to parse __all__ from {source_path}: {e}")
        
        if len(parts) > 1:
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        return parts, doc_path, full_doc_path, has_all
    
    def _parse_init_doc_and_all(self, source_path: Path) -> Tuple[Optional[str], List[str]]:
        """Parse __init__.py for module docstring and __all__ list using AST."""
        try:
            text = source_path.read_text(encoding='utf-8')
            tree = ast.parse(text)
            docstring = ast.get_docstring(tree)
            exports: List[str] = []
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == '__all__':
                            try:
                                value = ast.literal_eval(node.value)
                                if isinstance(value, (list, tuple)):
                                    exports = [str(v) for v in value]
                            except Exception:
                                pass
            return docstring, exports
        except Exception:
            return None, []

    def create_documentation_file(self, full_doc_path: Path, identifier: str, source_path: Path, package_path: str = "") -> bool:
        """Create documentation file"""
        try:
            with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                module_title = identifier.split('.')[-1] if '.' in identifier else identifier
                fd.write(f"# {module_title}\n\n")
                
                if source_path.name == '__init__.py':
                    # Special index page content sourced from __init__.py
                    # docstring, exports = self._parse_init_doc_and_all(source_path)
                    # if docstring:
                    #     fd.write(f"{docstring}\n\n")
                    # if exports:
                    #     fd.write("## Exports\n\n")
                    #     for name in exports:
                    #         fd.write(f"- `{name}`\n")
                    #     fd.write("\n")
                    # Also include mkdocstrings block for the package itself
                    # For bridgic.core.automa specifically, include worker decorator before the package block
                    if identifier == 'bridgic.core.automa':
                        fd.write("::: bridgic.core.automa.worker._worker_decorator\n")
                    fd.write(f"::: {identifier}\n")
                else:
                    if package_path:
                        description = self.config.get_package_description(package_path)
                        if description:
                            fd.write(f"> {description}\n\n")
                    fd.write(f"::: {identifier}\n")
                
            mkdocs_gen_files.set_edit_path(full_doc_path, source_path.relative_to(self.root))
            
            if self.config.verbose:
                logger.info(f"Generated documentation: {full_doc_path} -> {identifier}")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to create documentation file {full_doc_path}: {e}")
            self.error_files.append((source_path, str(e)))
            return False
    
    def process_package(self, package_path: str) -> None:
        """Process documentation generation for a single package"""
        code_src = self.root / package_path
        
        if not code_src.exists():
            logger.warning(f"Package path does not exist: {code_src}")
            return
            
        if not code_src.is_dir():
            logger.warning(f"Package path is not a directory: {code_src}")
            return
            
        logger.info(f"Processing package: {code_src}")
        
        python_files = []
        try:
            python_files = sorted([p for p in code_src.rglob("*.py") if self.is_valid_python_module(p)])
        except Exception as e:
            logger.error(f"Failed to scan package files {code_src}: {e}")
            return
            
        logger.info(f"Found {len(python_files)} Python files")
        
        # Track package nodes (directories with __init__.py)
        package_name = self.config.get_package_display_name(package_path)
        if package_name not in self.package_nodes:
            self.package_nodes[package_name] = {}

        for path in python_files:
            try:
                if self.should_exclude_path(path):
                    self.skipped_files.append(path)
                    if self.config.verbose:
                        logger.debug(f"Skipped file: {path}")
                    continue
                
                try:
                    module_path = path.relative_to(code_src).with_suffix("")
                except ValueError as e:
                    logger.error(f"Failed to calculate relative path {path}: {e}")
                    continue
                    
                doc_path = module_path.with_suffix(".md")
                full_doc_path = Path(self.config.docs_base_path) / package_name / doc_path

                parts = list(module_path.parts)

                if parts[-1] == "__init__":
                    parts, doc_path, full_doc_path, has_all = self.process_init_module(parts, doc_path, full_doc_path, path)
                    if not parts:
                        continue
                    # Only process __init__.py files that have __all__ defined
                    if not has_all:
                        if self.config.verbose:
                            logger.debug(f"Skipping {path} - no __all__ defined")
                        continue
                elif parts[-1] == "__main__":
                    continue
                elif self.config.only_index_pages:
                    # Skip non-__init__.py modules entirely when only generating index pages
                    continue
                
                # Generate correct module identifier (without package prefix)
                identifier = self.generate_module_identifier(parts)
                
                # Only add nav entries for index pages when only_index_pages is enabled
                if (not self.config.only_index_pages) or (self.config.only_index_pages and path.name == '__init__.py'):
                    nav_key = [package_name] + parts
                    nav_path = f"{package_name}/{doc_path.as_posix()}"
                    self.nav[nav_key] = nav_path
                
                if self.create_documentation_file(full_doc_path, identifier, path, package_path):
                    self.generated_files.append(full_doc_path)
                    
                    # Record package node only for index pages generated from __init__.py with __all__
                    if source_is_init := (path.name == '__init__.py' and has_all):
                        self.package_nodes[package_name][tuple(parts)] = f"{self.config.docs_base_path}/{package_name}/{doc_path.as_posix()}"
                    
            except Exception as e:
                logger.error(f"Failed to process file {path}: {e}")
                self.error_files.append((path, str(e)))
                continue

        # Build navigation structure for this package based on collected package nodes
        try:
            if self.package_nodes.get(package_name):
                if package_name not in self.nav_structure:
                    self.nav_structure[package_name] = {}

                nodes = self.package_nodes[package_name]
                # Group by first three parts (e.g., bridgic.core.automa)
                groups: Dict[Tuple[str, ...], Dict[str, Any]] = {}
                for parts_tuple, index_path in nodes.items():
                    if len(parts_tuple) >= 3:
                        key = parts_tuple[:3]
                        if key not in groups:
                            groups[key] = { 'Index': index_path, '__children__': {} }
                        # Track children mapping for later
                        groups[key]['__children__'][parts_tuple] = index_path

                # For each group, add immediate children (len == 4) as links; if deeper, nest under that child
                for first3, data in groups.items():
                    display_key = '.'.join(first3)
                    if display_key not in self.nav_structure[package_name]:
                        self.nav_structure[package_name][display_key] = []

                    # Ensure Index first
                    group_items: List[Any] = []
                    # Use the module name (last segment) as the label instead of a generic "Index"
                    group_label = first3[-1]
                    group_items.append({group_label: data['Index']})

                    # Build children under this group
                    child_index_map: Dict[Tuple[str, ...], str] = {}
                    for parts_tuple, index_path in nodes.items():
                        if len(parts_tuple) == 4 and parts_tuple[:3] == first3:
                            child_index_map[parts_tuple] = index_path

                    # Sort children by name
                    for child_parts in sorted(child_index_map.keys()):
                        child_name = child_parts[-1]
                        child_path = child_index_map[child_parts]

                        # Detect if this child has deeper subpackages
                        has_deeper = any((len(p) > 4 and list(p[:4]) == list(child_parts)) for p in nodes.keys())
                        if not has_deeper:
                            group_items.append({child_name: child_path})
                        else:
                            # Create collapsible child with its own Index and its immediate children (5th level)
                            # Label the child's index with the child module name instead of "Index"
                            child_items: List[Any] = [ {child_name: child_path} ]
                            grand_children = [p for p in nodes.keys() if (len(p) == 5 and list(p[:4]) == list(child_parts))]
                            for gc_parts in sorted(grand_children):
                                gc_name = gc_parts[-1]
                                gc_path = nodes[gc_parts]
                                child_items.append({gc_name: gc_path})
                            group_items.append({child_name: child_items})

                    # If there are no children besides Index, render as a single link instead of a collapsible group
                    if len(group_items) == 1:
                        if self.config.single_entry_as_group:
                            self.nav_structure[package_name][display_key] = group_items
                        else:
                            self.nav_structure[package_name][display_key] = data['Index']
                    else:
                        self.nav_structure[package_name][display_key] = group_items
        except Exception as e:
            logger.error(f"Failed to build nav structure for package {package_name}: {e}")
    
    def _build_nav_structure_only(self, package_path: str) -> None:
        """Build navigation structure without generating documentation files"""
        code_src = self.root / package_path
        
        if not code_src.exists():
            logger.warning(f"Package path does not exist: {code_src}")
            return
            
        if not code_src.is_dir():
            logger.warning(f"Package path is not a directory: {code_src}")
            return
            
        logger.info(f"Building navigation structure for package: {code_src}")
        
        python_files = []
        try:
            python_files = sorted([p for p in code_src.rglob("*.py") if self.is_valid_python_module(p)])
        except Exception as e:
            logger.error(f"Failed to scan package files {code_src}: {e}")
            return
            
        logger.info(f"Found {len(python_files)} Python files")
        
        # Track package nodes (directories with __init__.py)
        package_name = self.config.get_package_display_name(package_path)
        if package_name not in self.package_nodes:
            self.package_nodes[package_name] = {}

        for path in python_files:
            try:
                if self.should_exclude_path(path):
                    self.skipped_files.append(path)
                    if self.config.verbose:
                        logger.debug(f"Skipped file: {path}")
                    continue
                
                try:
                    module_path = path.relative_to(code_src).with_suffix("")
                except ValueError as e:
                    logger.error(f"Failed to calculate relative path {path}: {e}")
                    continue
                    
                doc_path = module_path.with_suffix(".md")
                full_doc_path = Path(self.config.docs_base_path) / package_name / doc_path

                parts = list(module_path.parts)

                if parts[-1] == "__init__":
                    parts, doc_path, full_doc_path, has_all = self.process_init_module(parts, doc_path, full_doc_path, path)
                    if not parts:
                        continue
                    # Only process __init__.py files that have __all__ defined
                    if not has_all:
                        if self.config.verbose:
                            logger.debug(f"Skipping {path} - no __all__ defined")
                        continue
                elif parts[-1] == "__main__":
                    continue
                elif self.config.only_index_pages:
                    # Skip non-__init__.py modules entirely when only generating index pages
                    continue
                
                # Record package node only for index pages generated from __init__.py with __all__
                if source_is_init := (path.name == '__init__.py' and has_all):
                    self.package_nodes[package_name][tuple(parts)] = f"{self.config.docs_base_path}/{package_name}/{doc_path.as_posix()}"
                    
            except Exception as e:
                logger.error(f"Failed to process file {path}: {e}")
                self.error_files.append((path, str(e)))
                continue

        # Build navigation structure for this package based on collected package nodes
        try:
            if self.package_nodes.get(package_name):
                if package_name not in self.nav_structure:
                    self.nav_structure[package_name] = {}

                nodes = self.package_nodes[package_name]
                # Group by first three parts (e.g., bridgic.core.automa)
                groups: Dict[Tuple[str, ...], Dict[str, Any]] = {}
                for parts_tuple, index_path in nodes.items():
                    if len(parts_tuple) >= 3:
                        key = parts_tuple[:3]
                        if key not in groups:
                            groups[key] = { 'Index': index_path, '__children__': {} }
                        # Track children mapping for later
                        groups[key]['__children__'][parts_tuple] = index_path

                # For each group, add immediate children (len == 4) as links; if deeper, nest under that child
                for first3, data in groups.items():
                    display_key = '.'.join(first3)
                    if display_key not in self.nav_structure[package_name]:
                        self.nav_structure[package_name][display_key] = []

                    # Ensure Index first
                    group_items: List[Any] = []
                    # Use the module name (last segment) as the label instead of a generic "Index"
                    group_label = first3[-1]
                    group_items.append({group_label: data['Index']})

                    # Build children under this group
                    child_index_map: Dict[Tuple[str, ...], str] = {}
                    for parts_tuple, index_path in nodes.items():
                        if len(parts_tuple) == 4 and parts_tuple[:3] == first3:
                            child_index_map[parts_tuple] = index_path

                    # Sort children by name
                    for child_parts in sorted(child_index_map.keys()):
                        child_name = child_parts[-1]
                        child_path = child_index_map[child_parts]

                        # Detect if this child has deeper subpackages
                        has_deeper = any((len(p) > 4 and list(p[:4]) == list(child_parts)) for p in nodes.keys())
                        if not has_deeper:
                            group_items.append({child_name: child_path})
                        else:
                            # Create collapsible child with its own Index and its immediate children (5th level)
                            # Label the child's index with the child module name instead of "Index"
                            child_items: List[Any] = [ {child_name: child_path} ]
                            grand_children = [p for p in nodes.keys() if (len(p) == 5 and list(p[:4]) == list(child_parts))]
                            for gc_parts in sorted(grand_children):
                                gc_name = gc_parts[-1]
                                gc_path = nodes[gc_parts]
                                child_items.append({gc_name: gc_path})
                            group_items.append({child_name: child_items})

                    # If there are no children besides Index, render as a single link instead of a collapsible group
                    if len(group_items) == 1:
                        if self.config.single_entry_as_group:
                            self.nav_structure[package_name][display_key] = group_items
                        else:
                            self.nav_structure[package_name][display_key] = data['Index']
                    else:
                        self.nav_structure[package_name][display_key] = group_items
        except Exception as e:
            logger.error(f"Failed to build nav structure for package {package_name}: {e}")
    
    def generate_package_index(self, package_path: str) -> None:
        """Generate index page for package"""
        if not self.config.generate_index_pages:
            return
            
        try:
            package_name = self.config.get_package_display_name(package_path)
            description = self.config.get_package_description(package_path)
            
            index_path = Path(self.config.docs_base_path) / package_name / "index.md"
            
            with mkdocs_gen_files.open(index_path, "w") as fd:
                fd.write(f"# {package_name}\n\n")
                
                if description:
                    fd.write(f"{description}\n\n")
                
                fd.write("## Module List\n\n")
                fd.write("This package contains the following modules and sub-packages:\n\n")
                
            logger.info(f"Generated package index page: {index_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate package index page {package_path}: {e}")

    def generate_summary(self) -> None:
        """Generate navigation summary file"""
        try:
            summary_path = f"{self.config.docs_base_path}/SUMMARY.md"
            with mkdocs_gen_files.open(summary_path, "w") as nav_file:
                nav_file.writelines(self.nav.build_literate_nav())
            logger.info(f"Generated navigation summary: {summary_path}")
        except Exception as e:
            logger.error(f"Failed to generate navigation summary: {e}")
    
    def print_statistics(self) -> None:
        """Print generation statistics"""
        logger.info("=" * 50)
        logger.info("Documentation generation statistics:")
        logger.info(f"Successfully generated: {len(self.generated_files)} files")
        logger.info(f"Skipped files: {len(self.skipped_files)} files")
        logger.info(f"Error files: {len(self.error_files)} files")
        
        if self.error_files and self.config.verbose:
            logger.info("\nError details:")
            for path, error in self.error_files:
                logger.error(f"  {path}: {error}")
        
        logger.info("=" * 50)
    
    def generate(self) -> None:
        """Execute the main documentation generation process"""
        logger.info("Starting API documentation generation...")
        logger.info(f"Working root directory: {self.root}")
        
        # Clean reference directory before generating new documentation
        self.clean_reference_directory()
        
        # First pass: build navigation structure without generating files
        for package_path in self.config.packages:
            self._build_nav_structure_only(package_path)
        
        # Generate the final mkdocs.yml from template
        if self.nav_structure:
            self.update_mkdocs_config()
        
        # # Ensure llms overview exists for nav index separation
        # try:
        #     overview_rel = Path(LLM_OVERVIEW_PATH)
        #     overview_abs = self.docs_dir / overview_rel
        #     if not overview_abs.exists():
        #         overview_abs.parent.mkdir(parents=True, exist_ok=True)
        #         with open(overview_abs, 'w', encoding='utf-8') as f:
        #             f.write("# LLM Integrations\n\nThis section provides an overview of available LLM backends.\n\n- bridgic.llms.openai\n- bridgic.llms.openai_like\n- bridgic.llms.vllm\n")
        #         logger.info(f"Created llms overview page: {overview_abs}")
        # except Exception as e:
        #     logger.warning(f"Failed to create llms overview page: {e}")
        
        # Second pass: generate documentation files
        for package_path in self.config.packages:
            self.process_package(package_path)
            # self.generate_package_index(package_path)
        
        # self.generate_summary()
        self.print_statistics()
        
        logger.info("Documentation generation completed!")
    
    def update_mkdocs_config(self) -> None:
        """Update the API Reference section of MkDocs configuration file"""
        try:
            mkdocs_path = self.docs_dir / "mkdocs.yml"
            updater = SafeMkDocsConfigUpdater(mkdocs_path)
            
            if self.nav_structure:
                success = updater.update_mkdocs_config(self.nav_structure)
                if success:
                    logger.info("Successfully updated MkDocs configuration file")
                else:
                    logger.warning("Failed to update MkDocs configuration file")
            else:
                logger.info("No generated documentation, skipping MkDocs configuration update")
                
        except Exception as e:
            logger.error(f"Error occurred while updating MkDocs configuration: {e}")

def main():
    """Main function"""
    try:
        # ensure the working directory is set to docs directory, so mkdocs_gen_files can find mkdocs.yml
        script_dir = Path(__file__).parent  # docs/scripts/
        docs_dir = script_dir.parent  # docs/
        original_cwd = Path.cwd()
        
        logger.info(f"Changing working directory from {original_cwd} to {docs_dir}")
        os.chdir(docs_dir)
        
        try:
            config = DocumentationConfig()
            generator = DocumentationGenerator(config)
            generator.generate()
        finally:
            # restore the original working directory
            os.chdir(original_cwd)
            logger.info(f"Restored working directory to {original_cwd}")
        
    except KeyboardInterrupt:
        logger.info("User interrupted operation")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error occurred during documentation generation: {e}")
        import traceback
        logger.debug(f"Detailed error information: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
