"""
tests/test_phase3_negotiation.py
==================================
Unit tests for NegotiationSystem (Phase 3 rewrite).

Tests the full Proposal lifecycle:
    create → pending_for / pending_from → accept / reject → tick (TTL)
Anti-spam constraints: max_pending_proposals, proposal_cooldown.
"""

import pytest

from district_accord.core.negotiation import NegotiationSystem, Proposal
from district_accord.utils.config import EnvConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_sys(
    max_pending: int = 3,
    ttl: int = 3,
    cooldown: int = 2,
) -> NegotiationSystem:
    cfg = EnvConfig(
        max_pending_proposals=max_pending,
        proposal_ttl=ttl,
        proposal_cooldown=cooldown,
    )
    sys = NegotiationSystem(cfg)
    sys.reset(4)
    return sys


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_no_proposals_on_reset(self):
        sys = make_sys()
        assert sys.all_pending() == []

    def test_reset_clears_proposals(self):
        sys = make_sys()
        p = sys.create(0, 1, "coalition", {}, current_turn=0)
        assert p is not None
        sys.reset(4)
        assert sys.all_pending() == []

    def test_reset_clears_cooldowns(self):
        sys = make_sys(cooldown=10)
        sys.create(0, 1, "coalition", {}, current_turn=0)
        sys.reset(4)
        # After reset, cooldown should be gone — can create immediately
        p = sys.create(0, 1, "coalition", {}, current_turn=0)
        assert p is not None


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

class TestCreate:
    def test_proposal_returned_on_success(self):
        sys = make_sys()
        p = sys.create(0, 1, "coalition", {}, current_turn=0)
        assert isinstance(p, Proposal)
        assert p.proposer == 0
        assert p.target == 1
        assert p.kind == "coalition"
        assert p.ttl == 3  # default ttl

    def test_proposal_id_increments(self):
        sys = make_sys()
        p0 = sys.create(0, 1, "coalition", {}, current_turn=0)
        p1 = sys.create(1, 2, "coalition", {}, current_turn=0)
        assert p0 is not None and p1 is not None
        assert p1.proposal_id == p0.proposal_id + 1

    def test_self_proposal_returns_none(self):
        sys = make_sys()
        p = sys.create(0, 0, "coalition", {}, current_turn=0)
        assert p is None

    def test_terms_stored(self):
        sys = make_sys()
        p = sys.create(0, 1, "share", {"amount": 0.2}, current_turn=0)
        assert p is not None
        assert p.terms == {"amount": 0.2}

    def test_appears_in_pending_for(self):
        sys = make_sys()
        sys.create(0, 1, "coalition", {}, current_turn=0)
        pending = sys.pending_for(1)
        assert len(pending) == 1
        assert pending[0].proposer == 0

    def test_appears_in_pending_from(self):
        sys = make_sys()
        sys.create(0, 1, "coalition", {}, current_turn=0)
        sent = sys.pending_from(0)
        assert len(sent) == 1
        assert sent[0].target == 1


# ---------------------------------------------------------------------------
# Anti-spam: max_pending_proposals
# ---------------------------------------------------------------------------

class TestMaxPending:
    def test_max_pending_blocks_creation(self):
        sys = make_sys(max_pending=2)
        # Agent 0 proposes to 1 and 2
        p0 = sys.create(0, 1, "coalition", {}, current_turn=0)
        p1 = sys.create(0, 2, "coalition", {}, current_turn=0)
        # Third proposal should be blocked
        p2 = sys.create(0, 3, "coalition", {}, current_turn=0)
        assert p0 is not None
        assert p1 is not None
        assert p2 is None

    def test_max_pending_allows_after_acceptance(self):
        sys = make_sys(max_pending=1)
        p = sys.create(0, 1, "coalition", {}, current_turn=0)
        assert p is not None
        # Accept the proposal → slot freed
        sys.accept(p.proposal_id, 1)
        # Now agent 0 can send another (to a different target, past cooldown)
        p2 = sys.create(0, 2, "coalition", {}, current_turn=100)
        assert p2 is not None


# ---------------------------------------------------------------------------
# Anti-spam: cooldown
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_cooldown_blocks_immediate_re_proposal(self):
        sys = make_sys(cooldown=3)
        sys.create(0, 1, "coalition", {}, current_turn=0)
        # On same turn, cooldown blocks retry to same target
        p2 = sys.create(0, 1, "coalition", {}, current_turn=0)
        assert p2 is None

    def test_cooldown_expires(self):
        sys = make_sys(cooldown=3)
        sys.create(0, 1, "coalition", {}, current_turn=0)
        # cooldown_end = 0 + 3 = 3; at turn 3, cooldown has ended
        p2 = sys.create(0, 1, "coalition", {}, current_turn=3)
        assert p2 is not None

    def test_cooldown_does_not_affect_different_target(self):
        sys = make_sys(cooldown=10)
        sys.create(0, 1, "coalition", {}, current_turn=0)
        # Same proposer, different target — no cooldown
        p2 = sys.create(0, 2, "coalition", {}, current_turn=0)
        assert p2 is not None


# ---------------------------------------------------------------------------
# TTL expiration
# ---------------------------------------------------------------------------

class TestTTL:
    def test_proposal_exists_before_expiry(self):
        sys = make_sys(ttl=3)
        sys.create(0, 1, "coalition", {}, current_turn=0)
        sys.tick()  # ttl=2
        sys.tick()  # ttl=1
        assert len(sys.pending_for(1)) == 1

    def test_proposal_expires_after_ttl_ticks(self):
        sys = make_sys(ttl=3)
        sys.create(0, 1, "coalition", {}, current_turn=0)
        expired = []
        for _ in range(3):
            expired.extend(sys.tick())
        assert len(expired) == 1
        assert expired[0].proposer == 0
        assert len(sys.pending_for(1)) == 0

    def test_expired_proposals_returned_by_tick(self):
        sys = make_sys(ttl=2)
        p = sys.create(0, 1, "coalition", {}, current_turn=0)
        sys.tick()
        expired = sys.tick()  # expires here
        assert len(expired) == 1
        assert expired[0].proposal_id == p.proposal_id


# ---------------------------------------------------------------------------
# Accept
# ---------------------------------------------------------------------------

class TestAccept:
    def test_accept_returns_proposal(self):
        sys = make_sys()
        p = sys.create(0, 1, "coalition", {}, current_turn=0)
        accepted = sys.accept(p.proposal_id, 1)
        assert accepted is not None
        assert accepted.proposer == 0
        assert accepted.target == 1

    def test_accept_removes_from_pending(self):
        sys = make_sys()
        p = sys.create(0, 1, "coalition", {}, current_turn=0)
        sys.accept(p.proposal_id, 1)
        assert len(sys.pending_for(1)) == 0

    def test_accept_wrong_target_fails(self):
        sys = make_sys()
        p = sys.create(0, 1, "coalition", {}, current_turn=0)
        accepted = sys.accept(p.proposal_id, 2)  # wrong target
        assert accepted is None

    def test_accept_nonexistent_proposal_returns_none(self):
        sys = make_sys()
        assert sys.accept(999, 1) is None

    def test_accept_first_returns_first_proposal(self):
        sys = make_sys()
        p0 = sys.create(0, 2, "coalition", {}, current_turn=0)
        p1 = sys.create(1, 2, "coalition", {}, current_turn=0)
        accepted = sys.accept_first(target=2, kind="coalition")
        assert accepted is not None
        assert accepted.proposal_id == p0.proposal_id  # sorted by id, first wins

    def test_accept_first_no_proposal_returns_none(self):
        sys = make_sys()
        assert sys.accept_first(target=0, kind="coalition") is None


# ---------------------------------------------------------------------------
# Reject
# ---------------------------------------------------------------------------

class TestReject:
    def test_reject_returns_true_on_success(self):
        sys = make_sys()
        p = sys.create(0, 1, "coalition", {}, current_turn=0)
        result = sys.reject(p.proposal_id, 1)
        assert result is True

    def test_reject_removes_from_pending(self):
        sys = make_sys()
        p = sys.create(0, 1, "coalition", {}, current_turn=0)
        sys.reject(p.proposal_id, 1)
        assert len(sys.pending_for(1)) == 0

    def test_reject_wrong_target_fails(self):
        sys = make_sys()
        p = sys.create(0, 1, "coalition", {}, current_turn=0)
        assert sys.reject(p.proposal_id, 2) is False

    def test_reject_nonexistent_returns_false(self):
        sys = make_sys()
        assert sys.reject(999, 0) is False

    def test_reject_first_helper(self):
        sys = make_sys()
        p = sys.create(0, 1, "coalition", {}, current_turn=0)
        rejected = sys.reject_first(target=1, kind="coalition")
        assert rejected is not None
        assert rejected.proposal_id == p.proposal_id
        assert len(sys.pending_for(1)) == 0


# ---------------------------------------------------------------------------
# Aid request helpers
# ---------------------------------------------------------------------------

class TestAidHelpers:
    def test_has_active_request_true(self):
        sys = make_sys()
        sys.create(0, 1, "aid", {}, current_turn=0)
        assert sys.has_active_request(0) is True

    def test_has_active_request_false_after_accept(self):
        sys = make_sys()
        p = sys.create(0, 1, "aid", {}, current_turn=0)
        sys.accept(p.proposal_id, 1)
        assert sys.has_active_request(0) is False

    def test_active_requesters_set(self):
        sys = make_sys()
        sys.create(0, 1, "aid", {}, current_turn=0)
        sys.create(2, 3, "aid", {}, current_turn=0)
        assert sys.active_requesters() == {0, 2}

    def test_coalition_proposers_for(self):
        sys = make_sys()
        sys.create(0, 2, "coalition", {}, current_turn=0)
        sys.create(1, 2, "coalition", {}, current_turn=0)
        assert sys.coalition_proposers_for(2) == {0, 1}

    def test_coalition_proposers_excludes_aid(self):
        sys = make_sys()
        sys.create(0, 2, "aid", {}, current_turn=0)  # aid, not coalition
        assert sys.coalition_proposers_for(2) == set()


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------

class TestToDict:
    def test_to_dict_structure(self):
        sys = make_sys()
        sys.create(0, 1, "coalition", {}, current_turn=0)
        d = sys.to_dict()
        assert "pending_count" in d
        assert "proposals" in d
        assert d["pending_count"] == 1
