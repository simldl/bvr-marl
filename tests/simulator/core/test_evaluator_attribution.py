from bvr_marl_core.simulator.simulator import Simulator


def test_contact_truth_lineage_is_evaluator_owned_and_ambiguity_fails_closed():
    sim = Simulator(random_seed=7)
    sim.register_sensor_report_truth_association("radar-blue-1", 10, 101)
    sim.register_sensor_report_truth_association("radar-blue-1", 11, 101)
    sim.register_sensor_report_truth_association("radar-blue-1", 12, 202)

    sim.refresh_contact_truth_associations(
        "radar-blue-1",
        {
            "track-a": (("radar-blue-1", 10), ("radar-blue-1", 11)),
            "track-ambiguous": (("radar-blue-1", 11), ("radar-blue-1", 12)),
        },
    )

    assert sim.evaluator_truth_id_for_contact("radar-blue-1", "track-a") == 101
    assert sim.evaluator_truth_id_for_contact("radar-blue-1", "track-ambiguous") is None


def test_contact_truth_lineage_is_removed_when_contact_disappears():
    sim = Simulator(random_seed=7)
    sim.register_sensor_report_truth_association("radar-blue-1", 10, 101)
    sim.refresh_contact_truth_associations("radar-blue-1", {"track-a": (("radar-blue-1", 10),)})
    assert sim.evaluator_truth_id_for_contact("radar-blue-1", "track-a") == 101

    sim.refresh_contact_truth_associations("radar-blue-1", {})

    assert sim.evaluator_truth_id_for_contact("radar-blue-1", "track-a") is None


def test_datalink_only_contact_uses_original_report_source_for_attribution():
    sim = Simulator(random_seed=7)
    sim.register_sensor_report_truth_association("sender-radar", 1, 101)
    sim.register_sensor_report_truth_association("receiver-radar", 1, 202)

    sim.refresh_contact_truth_associations(
        "receiver-radar", {"datalink-track": (("sender-radar", 1),)}
    )

    assert sim.evaluator_truth_id_for_contact("receiver-radar", "datalink-track") == 101


def test_attribution_survives_a_single_mis_associated_report():
    """One contaminating report must not suppress a track's attribution.

    Track lineage only ever grows, so requiring unanimity meant one report that
    association placed on the wrong aircraft disabled attribution for the rest of the
    episode -- and with it every hit check for weapons fired at that track.
    """
    sim = Simulator(random_seed=7)
    for report_id in range(1, 6):
        sim.register_sensor_report_truth_association("radar-blue-1", report_id, 101)
    sim.register_sensor_report_truth_association("radar-blue-1", 6, 202)

    lineage = tuple(("radar-blue-1", report_id) for report_id in range(1, 7))
    sim.refresh_contact_truth_associations("radar-blue-1", {"track-a": lineage})

    assert sim.evaluator_truth_id_for_contact("radar-blue-1", "track-a") == 101


def test_attribution_follows_a_track_that_re_associates_onto_another_aircraft():
    """Only recent evidence votes, so stale history cannot outvote the present."""
    sim = Simulator(random_seed=7)
    old = range(1, 21)
    new = range(21, 31)
    for report_id in old:
        sim.register_sensor_report_truth_association("radar-blue-1", report_id, 101)
    for report_id in new:
        sim.register_sensor_report_truth_association("radar-blue-1", report_id, 202)

    lineage = tuple(("radar-blue-1", report_id) for report_id in [*old, *new])
    sim.refresh_contact_truth_associations("radar-blue-1", {"track-a": lineage})

    assert sim.evaluator_truth_id_for_contact("radar-blue-1", "track-a") == 202


def test_evenly_split_evidence_is_still_reported_as_ambiguous():
    sim = Simulator(random_seed=7)
    sim.register_sensor_report_truth_association("radar-blue-1", 1, 101)
    sim.register_sensor_report_truth_association("radar-blue-1", 2, 202)

    sim.refresh_contact_truth_associations(
        "radar-blue-1",
        {"track-a": (("radar-blue-1", 1), ("radar-blue-1", 2))},
    )

    assert sim.evaluator_truth_id_for_contact("radar-blue-1", "track-a") is None
