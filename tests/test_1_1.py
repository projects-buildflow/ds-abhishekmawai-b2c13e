"""Tests for Task 1.1: Environment Setup

Objective: Set up local development environment and verify everything works.

This test verifies that:
1. Student ran the verification script successfully
2. Environment has correct Python version (3.8+)
3. Required packages are installed (pandas, numpy, pytest)
4. Git is configured with name and email
"""

import pytest
import os
import re
from pathlib import Path


class TestSetupFileExists:
    """Verify the setup verification output file exists."""

    def test_setup_complete_file_exists(self, student_folder):
        """Student must create setup_complete.txt with verification output."""
        if not student_folder:
            pytest.skip("Student folder not provided")

        setup_path = Path(student_folder) / "week-1" / "setup_complete.txt"
        assert setup_path.exists(), (
            f"setup_complete.txt not found at {setup_path}\n\n"
            "To complete this task:\n"
            "1. Run: python scripts/verify_setup.py\n"
            "2. Copy the output to: cohort/{your-username}/week-1/setup_complete.txt"
        )

    def test_setup_file_not_empty(self, student_folder):
        """Setup file must have meaningful content."""
        if not student_folder:
            pytest.skip("Student folder not provided")

        setup_path = Path(student_folder) / "week-1" / "setup_complete.txt"
        if not setup_path.exists():
            pytest.skip("setup_complete.txt not found")

        content = setup_path.read_text().strip()
        assert len(content) >= 50, (
            f"setup_complete.txt has only {len(content)} characters.\n"
            "Expected the full output from verify_setup.py script."
        )


class TestPythonVersion:
    """Verify correct Python version is installed."""

    def test_has_python_version(self, student_folder):
        """Setup file must contain Python version."""
        if not student_folder:
            pytest.skip("Student folder not provided")

        setup_path = Path(student_folder) / "week-1" / "setup_complete.txt"
        if not setup_path.exists():
            pytest.skip("setup_complete.txt not found")

        content = setup_path.read_text()

        # Look for Python version pattern (e.g., "Python 3.11.4" or "3.11")
        version_pattern = r"[Pp]ython\s*:?\s*(\d+\.\d+)"
        match = re.search(version_pattern, content)

        assert match, (
            "Could not find Python version in setup_complete.txt.\n"
            "Expected format like 'Python: 3.11.4' or 'Python 3.11'"
        )

    def test_python_version_meets_minimum(self, student_folder):
        """Python version must be 3.8 or higher."""
        if not student_folder:
            pytest.skip("Student folder not provided")

        setup_path = Path(student_folder) / "week-1" / "setup_complete.txt"
        if not setup_path.exists():
            pytest.skip("setup_complete.txt not found")

        content = setup_path.read_text()

        # Extract version numbers
        version_pattern = r"[Pp]ython\s*:?\s*(\d+)\.(\d+)"
        match = re.search(version_pattern, content)

        if not match:
            pytest.skip("Could not parse Python version")

        major, minor = int(match.group(1)), int(match.group(2))

        assert major >= 3 and minor >= 8, (
            f"Python version {major}.{minor} is too old.\n"
            "This project requires Python 3.8 or higher.\n"
            "Please install a newer version from python.org"
        )


class TestRequiredPackages:
    """Verify required packages are installed."""

    def test_has_pandas(self, student_folder):
        """Pandas must be installed and documented."""
        if not student_folder:
            pytest.skip("Student folder not provided")

        setup_path = Path(student_folder) / "week-1" / "setup_complete.txt"
        if not setup_path.exists():
            pytest.skip("setup_complete.txt not found")

        content = setup_path.read_text().lower()

        assert "pandas" in content, (
            "pandas not found in setup_complete.txt.\n"
            "Make sure pandas is installed: pip install pandas\n"
            "Then run verify_setup.py again."
        )

    def test_has_numpy(self, student_folder):
        """NumPy must be installed and documented."""
        if not student_folder:
            pytest.skip("Student folder not provided")

        setup_path = Path(student_folder) / "week-1" / "setup_complete.txt"
        if not setup_path.exists():
            pytest.skip("setup_complete.txt not found")

        content = setup_path.read_text().lower()

        assert "numpy" in content, (
            "numpy not found in setup_complete.txt.\n"
            "Make sure numpy is installed: pip install numpy\n"
            "Then run verify_setup.py again."
        )

    def test_has_pytest(self, student_folder):
        """pytest must be installed and documented."""
        if not student_folder:
            pytest.skip("Student folder not provided")

        setup_path = Path(student_folder) / "week-1" / "setup_complete.txt"
        if not setup_path.exists():
            pytest.skip("setup_complete.txt not found")

        content = setup_path.read_text().lower()

        assert "pytest" in content, (
            "pytest not found in setup_complete.txt.\n"
            "Make sure pytest is installed: pip install pytest\n"
            "Then run verify_setup.py again."
        )


class TestGitConfiguration:
    """Verify Git is properly configured."""

    def test_has_git_configured(self, student_folder):
        """Git configuration should be mentioned."""
        if not student_folder:
            pytest.skip("Student folder not provided")

        setup_path = Path(student_folder) / "week-1" / "setup_complete.txt"
        if not setup_path.exists():
            pytest.skip("setup_complete.txt not found")

        content = setup_path.read_text().lower()

        # Look for git configuration indicators
        has_git = "git" in content and ("configured" in content or "user" in content or "name" in content)

        assert has_git, (
            "Git configuration not found in setup_complete.txt.\n"
            "Make sure Git is configured:\n"
            "  git config user.name 'Your Name'\n"
            "  git config user.email 'your@email.com'\n"
            "Then run verify_setup.py again."
        )


class TestVerificationToken:
    """Verify the setup token was generated."""

    def test_has_verification_token(self, student_folder):
        """Setup file should contain the CARTLY verification token."""
        if not student_folder:
            pytest.skip("Student folder not provided")

        setup_path = Path(student_folder) / "week-1" / "setup_complete.txt"
        if not setup_path.exists():
            pytest.skip("setup_complete.txt not found")

        content = setup_path.read_text()

        # Look for token pattern: CARTLY-XXXX-XXXX or similar
        token_pattern = r"CARTLY-[A-Z0-9]+-[A-Z0-9]+"
        has_token = bool(re.search(token_pattern, content))

        assert has_token, (
            "Verification token not found in setup_complete.txt.\n"
            "The token looks like: CARTLY-XXXXXXXXXXXX-XXXX\n"
            "Run 'python scripts/verify_setup.py' and copy the full output."
        )

    def test_all_checks_passed(self, student_folder):
        """All verification checks should have passed."""
        if not student_folder:
            pytest.skip("Student folder not provided")

        setup_path = Path(student_folder) / "week-1" / "setup_complete.txt"
        if not setup_path.exists():
            pytest.skip("setup_complete.txt not found")

        content = setup_path.read_text().lower()

        # Check for failure indicators
        has_failures = "failed" in content or "error" in content or "not found" in content

        # Check for success indicators
        has_success = "passed" in content or "success" in content or "✓" in content or "complete" in content

        assert has_success and not has_failures, (
            "Some verification checks may have failed.\n"
            "Review the output and fix any issues before proceeding."
        )
