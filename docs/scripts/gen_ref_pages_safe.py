"""
Safe API documentation generation script
Uses staging and recovery mechanism to protect other configurations
"""

import logging
import sys
import yaml
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any
import re

import mkdocs_gen_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentationConfig:
    """Documentation generation configuration class"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Set default configuration
        self._set_defaults()
        
        # Load configuration file if provided
        if config_file:
            self.load_from_file(config_file)
        else:
            # Try to load default configuration file
            default_config = Path(__file__).parent / "doc_config.yaml"
            if default_config.exists():
                self.load_from_file(str(default_config))
    
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
        self.show_source = True
        self.show_root_heading = True
        self.show_root_toc_entry = True
        self.generate_index_pages = True
        self.docstring_style = "numpy"
        
        # mkdocstrings选项
        self.mkdocstrings_options = {
            "docstring_options": {"ignore_init_summary": True},
            "filters": ["!^_", "!^__init__"],
            "members_order": "source",
            "merge_init_into_class": False,
            "separate_signature": True,
            "signature_crossrefs": True
        }
    
    def load_from_file(self, config_file: str):
        """Load configuration from YAML configuration file"""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                logger.warning(f"Configuration file does not exist: {config_file}")
                return
                
            with open(config_path, 'r', encoding='utf-8') as f:
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
            
            # Update mkdocstrings options
            if 'mkdocstrings_options' in config_data:
                self.mkdocstrings_options.update(config_data['mkdocstrings_options'])
                    
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
        """Safely update the API Reference section of mkdocs.yml, protecting other configurations"""
        try:
            if not self.mkdocs_path.exists():
                logger.error(f"MkDocs configuration file does not exist: {self.mkdocs_path}")
                return False
            
            # Read original file content
            with open(self.mkdocs_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            logger.debug("Starting safe MkDocs configuration update...")
            
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
                nav_content = remaining_content[:next_config_match.start() + 1]  # 保留换行符
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
            
            logger.info("Successfully updated MkDocs configuration file, preserving all other configurations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update MkDocs configuration file: {e}")
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
        api_nav = [{'Index': 'api/index.md'}]
        
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
                formatted_name = self._format_display_name(key)
                nav_items.append({formatted_name: sub_nav})
            else:
                formatted_name = self._format_display_name(key)
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
                        lines.append(f"{base_indent}    - {key}:")
                        sub_yaml = self._nav_to_yaml_string(value, indent + 2)
                        if sub_yaml:
                            lines.append(sub_yaml)
                    else:
                        lines.append(f"{base_indent}    - {key}: {value}")
            else:
                lines.append(f"{base_indent}    - {item}")
        
        return "\n".join(lines)

class DocumentationGenerator:
    """Main documentation generator class"""
    
    def __init__(self, config: DocumentationConfig):
        self.config = config
        self.nav = mkdocs_gen_files.Nav()
        self.root = Path(__file__).parent.parent.parent
        self.generated_files = []
        self.skipped_files = []
        self.error_files = []
        self.nav_structure = {}
        
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
    
    def process_init_module(self, parts: List[str], doc_path: Path, full_doc_path: Path) -> Tuple[List[str], Path, Path]:
        """Process __init__.py module"""
        if len(parts) > 1:
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        return parts, doc_path, full_doc_path
    
    def create_documentation_file(self, full_doc_path: Path, identifier: str, source_path: Path, package_path: str = "") -> bool:
        """Create documentation file"""
        try:
            with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                module_title = identifier.split('.')[-1] if '.' in identifier else identifier
                fd.write(f"# {module_title}\n\n")
                
                if package_path:
                    description = self.config.get_package_description(package_path)
                    if description:
                        fd.write(f"> {description}\n\n")
                
                fd.write(f"::: {identifier}\n")
                fd.write("    options:\n")
                fd.write(f"      show_source: {str(self.config.show_source).lower()}\n")
                fd.write(f"      show_root_heading: {str(self.config.show_root_heading).lower()}\n")
                fd.write(f"      show_root_toc_entry: {str(self.config.show_root_toc_entry).lower()}\n")
                
                if self.config.mkdocstrings_options:
                    for key, value in self.config.mkdocstrings_options.items():
                        if key == "docstring_options" and isinstance(value, dict):
                            fd.write(f"      {key}:\n")
                            for sub_key, sub_value in value.items():
                                fd.write(f"        {sub_key}: {str(sub_value).lower()}\n")
                        elif key == "filters" and isinstance(value, list):
                            fd.write(f"      {key}:\n")
                            for filter_item in value:
                                fd.write(f"        - \"{filter_item}\"\n")
                        elif isinstance(value, bool):
                            fd.write(f"      {key}: {str(value).lower()}\n")
                        elif isinstance(value, str):
                            fd.write(f"      {key}: \"{value}\"\n")
                        else:
                            fd.write(f"      {key}: {value}\n")
                
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
                package_name = self.config.get_package_display_name(package_path)
                full_doc_path = Path(self.config.docs_base_path) / package_name / doc_path

                parts = list(module_path.parts)

                if parts[-1] == "__init__":
                    parts, doc_path, full_doc_path = self.process_init_module(parts, doc_path, full_doc_path)
                    if not parts:
                        continue
                elif parts[-1] == "__main__":
                    continue
                
                # Generate correct module identifier (without package prefix)
                identifier = self.generate_module_identifier(parts)
                
                nav_key = [package_name] + parts
                nav_path = f"{package_name}/{doc_path.as_posix()}"
                self.nav[nav_key] = nav_path
                
                if self.create_documentation_file(full_doc_path, identifier, path, package_path):
                    self.generated_files.append(full_doc_path)
                    
                    # Collect navigation structure
                    if package_name not in self.nav_structure:
                        self.nav_structure[package_name] = {}
                    
                    current_nav = self.nav_structure[package_name]
                    for part in parts[:-1]:
                        if part not in current_nav:
                            current_nav[part] = {}
                        elif isinstance(current_nav[part], str):
                            current_nav[part] = {}
                        current_nav = current_nav[part]
                    
                    file_title = parts[-1] if parts else identifier.split('.')[-1]
                    if isinstance(current_nav, dict):
                        current_nav[file_title] = f"{self.config.docs_base_path}/{package_name}/{doc_path.as_posix()}"
                    
            except Exception as e:
                logger.error(f"Failed to process file {path}: {e}")
                self.error_files.append((path, str(e)))
                continue
    
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
        
        for package_path in self.config.packages:
            self.process_package(package_path)
            self.generate_package_index(package_path)
        
        self.generate_summary()
        self.update_mkdocs_config()
        self.print_statistics()
        
        logger.info("Documentation generation completed!")
    
    def update_mkdocs_config(self) -> None:
        """Update the API Reference section of MkDocs configuration file"""
        try:
            mkdocs_path = self.root / "docs" / "mkdocs.yml"
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
        config = DocumentationConfig()
        generator = DocumentationGenerator(config)
        generator.generate()
        
    except KeyboardInterrupt:
        logger.info("User interrupted operation")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error occurred during documentation generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
