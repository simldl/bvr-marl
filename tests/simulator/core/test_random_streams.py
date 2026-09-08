from bvr_marl_core.simulator.core.random_streams import EpisodeRandomStreams
from bvr_marl_core.simulator.simulator import Simulator


def test_named_streams_reproduce_independently_of_request_order():
    first = EpisodeRandomStreams(42)
    radar_expected = first.generator("radar", 7).random(5)
    link_expected = first.generator("datalink_link", "7->8").random(5)

    second = EpisodeRandomStreams(42)
    link_actual = second.generator("datalink_link", "7->8").random(5)
    radar_actual = second.generator("radar", 7).random(5)

    assert (radar_actual == radar_expected).all()
    assert (link_actual == link_expected).all()


def test_entity_ids_receive_distinct_streams():
    streams = EpisodeRandomStreams(42)
    assert streams.generator("radar", 1).random() != streams.generator("radar", 2).random()


def test_simulator_reseed_recreates_all_named_streams_and_metadata():
    simulator = Simulator(random_seed=1)
    simulator.seed(77)
    expected = simulator.random_streams.generator("passive_rf", "1:2").random(4)

    simulator.seed(77)
    actual = simulator.random_streams.generator("passive_rf", "1:2").random(4)

    assert (actual == expected).all()
    assert simulator.replay_metadata["root_seed"] == 77
    assert simulator.replay_metadata["stream_schema_version"] == 1
