"""
tests/test_phase3_coalition.py
================================
Unit tests for CoalitionSystem (Phase 3 rewrite).

Tests the new API: new_coalition / join / leave / same_coalition / coalition_size / is_full.
No pending_invites (moved to NegotiationSystem).
"""

import pytest

from district_accord.core.coalition import CoalitionSystem
from district_accord.utils.config import EnvConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_sys(num_districts: int = 4, max_coalition_size: int = 12) -> CoalitionSystem:
    cfg = EnvConfig(max_coalition_size=max_coalition_size)
    sys = CoalitionSystem(cfg)
    sys.reset(num_districts)
    return sys


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_all_unaffiliated_on_reset(self):
        sys = make_sys(4)
        for i in range(4):
            assert sys.memberships[i] is None

    def test_no_active_coalitions_on_reset(self):
        sys = make_sys(4)
        assert sys.active_coalitions() == {}

    def test_coalition_size_zero_when_solo(self):
        sys = make_sys(4)
        assert sys.coalition_size(0) == 0


# ---------------------------------------------------------------------------
# new_coalition / join
# ---------------------------------------------------------------------------

class TestFormation:
    def test_new_coalition_assigns_founder(self):
        sys = make_sys(4)
        cid = sys.new_coalition(0)
        assert sys.get_coalition(0) == cid
        assert sys.coalition_size(0) == 1

    def test_new_coalition_unique_ids(self):
        sys = make_sys(4)
        cid0 = sys.new_coalition(0)
        cid1 = sys.new_coalition(1)
        assert cid0 != cid1

    def test_join_adds_member(self):
        sys = make_sys(4)
        cid = sys.new_coalition(0)
        result = sys.join(1, cid)
        assert result is True
        assert sys.get_coalition(1) == cid
        assert sys.coalition_size(1) == 2

    def test_join_existing_coalition(self):
        sys = make_sys(4)
        cid = sys.new_coalition(0)
        sys.join(1, cid)
        sys.join(2, cid)
        assert sys.coalition_size(0) == 3
        assert sys.coalition_members(cid) == {0, 1, 2}

    def test_join_already_member_noop(self):
        sys = make_sys(4)
        cid = sys.new_coalition(0)
        sys.join(1, cid)
        result = sys.join(1, cid)  # join again
        assert result is True
        assert sys.coalition_size(0) == 2  # still 2, not 3

    def test_join_nonexistent_coalition_fails(self):
        sys = make_sys(4)
        result = sys.join(0, 999)
        assert result is False

    def test_join_switches_coalition(self):
        """Agent leaves old coalition when joining a new one."""
        sys = make_sys(4)
        cid0 = sys.new_coalition(0)
        cid1 = sys.new_coalition(1)
        sys.join(2, cid0)          # agent 2 joins coalition 0
        assert sys.get_coalition(2) == cid0
        sys.join(2, cid1)          # agent 2 switches to coalition 1
        assert sys.get_coalition(2) == cid1
        assert 2 not in sys.coalition_members(cid0)


# ---------------------------------------------------------------------------
# Max coalition size
# ---------------------------------------------------------------------------

class TestMaxSize:
    def test_max_size_enforcement(self):
        sys = make_sys(4, max_coalition_size=2)
        cid = sys.new_coalition(0)
        assert sys.join(1, cid) is True   # size → 2
        assert sys.join(2, cid) is False  # rejected: full
        assert sys.get_coalition(2) is None

    def test_is_full_true_at_limit(self):
        sys = make_sys(4, max_coalition_size=2)
        cid = sys.new_coalition(0)
        sys.join(1, cid)
        assert sys.is_full(cid) is True

    def test_is_full_false_below_limit(self):
        sys = make_sys(4, max_coalition_size=3)
        cid = sys.new_coalition(0)
        sys.join(1, cid)
        assert sys.is_full(cid) is False

    def test_is_full_false_for_unknown_coalition(self):
        sys = make_sys(4)
        assert sys.is_full(999) is False


# ---------------------------------------------------------------------------
# Leave
# ---------------------------------------------------------------------------

class TestLeave:
    def test_leave_removes_from_membership(self):
        sys = make_sys(4)
        cid = sys.new_coalition(0)
        sys.join(1, cid)
        result = sys.leave(1)
        assert result is True
        assert sys.get_coalition(1) is None
        assert 1 not in sys.coalition_members(cid)

    def test_leave_solo_agent_returns_false(self):
        sys = make_sys(4)
        assert sys.leave(0) is False

    def test_coalition_disbanded_when_last_member_leaves(self):
        sys = make_sys(4)
        cid = sys.new_coalition(0)
        sys.leave(0)
        assert cid not in sys.active_coalitions()

    def test_remaining_members_unaffected_after_leave(self):
        sys = make_sys(4)
        cid = sys.new_coalition(0)
        sys.join(1, cid)
        sys.join(2, cid)
        sys.leave(1)
        assert sys.coalition_size(0) == 2
        assert 1 not in sys.coalition_members(cid)
        assert {0, 2} == sys.coalition_members(cid)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

class TestQueries:
    def test_same_coalition_true(self):
        sys = make_sys(4)
        cid = sys.new_coalition(0)
        sys.join(1, cid)
        assert sys.same_coalition(0, 1) is True

    def test_same_coalition_false_different_coalitions(self):
        sys = make_sys(4)
        cid0 = sys.new_coalition(0)
        cid1 = sys.new_coalition(1)
        assert sys.same_coalition(0, 1) is False

    def test_same_coalition_false_one_solo(self):
        sys = make_sys(4)
        sys.new_coalition(0)
        assert sys.same_coalition(0, 1) is False

    def test_get_coalition_returns_none_for_solo(self):
        sys = make_sys(4)
        assert sys.get_coalition(0) is None

    def test_coalition_members_empty_for_unknown(self):
        sys = make_sys(4)
        assert sys.coalition_members(999) == set()

    def test_to_dict_structure(self):
        sys = make_sys(4)
        cid = sys.new_coalition(0)
        sys.join(1, cid)
        d = sys.to_dict()
        assert "memberships" in d
        assert "coalitions" in d
        assert str(cid) in d["coalitions"]
