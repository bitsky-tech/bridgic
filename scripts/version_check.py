#!/usr/bin/env python3
"""
Check version compatibility with target repository according to PEP 440.
"""

import re
import sys
import argparse
from pathlib import Path


def parse_version(version_str):
    """Parse version string and return version type"""
    # PEP 440 version specification
    # Development versions: 1.0.0.dev1, 1.0.0a1.dev1
    # Pre-release versions: 1.0.0a1, 1.0.0b1, 1.0.0rc1
    # Release versions: 1.0.0, 1.0.1
    # Post-release versions: 1.0.post1
    
    if re.search(r'\.dev\d+', version_str):
        return 'dev'
    elif re.search(r'\d+(a|alpha)\d+', version_str):
        return 'alpha'
    elif re.search(r'\d+(b|beta)\d+', version_str):
        return 'beta'
    elif re.search(r'\d+rc\d+', version_str):
        return 'rc'
    elif re.search(r'\.post\d+', version_str):
        return 'post'
    elif re.match(r'^\d+\.\d+(\.\d+)?$', version_str):
        return 'release'
    else:
        return 'unknown'


def check_version_repo_compatibility(version, repo):
    """Check version compatibility with repository"""
    version_type = parse_version(version)
    
    # Development versions can only be published to btsk-repo
    if version_type == 'dev':
        return repo == 'btsk'
    
    # Pre-release versions (alpha, beta, rc, post) can be published to pypi or testpypi
    if version_type in ['alpha', 'beta', 'rc', 'post']:
        return repo in ['pypi', 'testpypi']
    
    # Release versions can only be published to pypi
    if version_type == 'release':
        return repo == 'pypi'
    
    # Unknown version types are not allowed to be published
    return False


def main():
    parser = argparse.ArgumentParser(description='Check version compatibility with target repository')
    parser.add_argument('--version', required=True, help='Version string')
    parser.add_argument('--repo', required=True, help='Target repository (btsk, pypi, testpypi)')
    parser.add_argument('--package', help='Package name (for error messages)')
    
    args = parser.parse_args()
    
    if check_version_repo_compatibility(args.version, args.repo):
        print(f"✓ Version [{args.package}-{args.version}] is compatible with repository [{args.repo}].")
        sys.exit(0)
    else:
        print(f"✗ Version [{args.package}-{args.version}] is not compatible with repository [{args.repo}].")

        version_type = parse_version(args.version)
        if version_type == 'dev':
            print("  Development versions can only be published to [btsk].")
        elif version_type in ['alpha', 'beta', 'rc', 'post']:
            print("  Pre-release versions can only be published to [pypi] or [testpypi].")
        elif version_type == 'release':
            print("  Release versions can only be published to [pypi].")
        else:
            print("  Unknown version type.")
        
        sys.exit(1)


if __name__ == '__main__':
    main()