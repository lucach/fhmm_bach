import argparse
import constants
import math
from statistics import mean
from factorial_hmm_lib import *
from hmmlearn import hmm
from music21_helpers import *

def init_random_fhmm(n_steps, observed_alphabet_size, K, M):
    D = observed_alphabet_size  # observable states' alphabet size
    params = {
        'hidden_alphabet_size': K,
        'n_hidden_chains': M,
        'observed_alphabet_size': D,
        'n_observed_chains': 1,
    }
    params['initial_hidden_state'] = np.zeros((M, K))
    params['transition_matrices'] = np.zeros((M, K, K))
    params['obs_given_hidden'] = np.zeros([K] * M + [D])
    for i in range(M):
        # Uniform transition probability (1/K towards every state)
        params['transition_matrices'][i, :, :] = [[1 / K] * K] * K
        # Uniform initial state probability (1/K for every state)
        params['initial_hidden_state'][i, :] = [1 / K] * K
    for st in itertools.product(*[range(K)] * M):
        # Uniform emission probability (1/D)
        params['obs_given_hidden'][list(st) + [Ellipsis]] = 1 / D

    return FullDiscreteFactorialHMM(params=params, n_steps=n_steps,
                                    calculate_on_init=True)


def load_songs_from_pickle(pickle_filename):
    with open("dataset/%s.pickle" % pickle_filename, "rb") as f:
        return pickle.load(f)


# Init function to load datasets.
def init():
    # Load datasets
    authors = constants.AUTHORS
    songs = {}
    for author in authors:
        songs[author] = load_songs_from_pickle(author)

    # Hard limit songs to be within MAX_STEPS limit.
    for author in songs:
        songs[author] = [song[:constants.MAX_STEPS] for song in songs[author]]

    # List notes' durations found in the dataset.
    durations_list = constants.ALL_DURATIONS

    # List possible values for "pitch space" (integers only to ignore
    # microtones).
    ps_list = list(range(constants.MIN_PS, constants.MAX_PS + 1))

    # Build a list with every possible combination of (duration, ps) by doing
    # a cartesian product. Also, build a dict to obtain in constant time the
    # index of a note.
    events_list = list(itertools.product(*[durations_list, ps_list]))
    events_codes = {}
    for i in range(len(events_list)):
        events_codes[events_list[i]] = i

    return songs, events_codes, events_list


# Returns an integer which maps one-to-one to a note, using "events_codes".
def get_repr(events_codes, event):
    event_data = (event["duration"], event["ps"])
    return events_codes[event_data]


def get_nearest_value(value, l):
    return min(l, key=lambda x: abs(x - value))


def approximate_states(new_states, original_states, events_list,
                       events_codes, adapt_for_hmmlearn=False):
    original_states_flat = [state for state
                            in list(itertools.chain(*original_states))]
    if adapt_for_hmmlearn:
        original_states_flat = [state[0] for state in original_states_flat]

    original_notes = [events_list[state] for state in original_states_flat]
    durations_seen = set()
    ps_seen = set()
    for note in original_notes:
        durations_seen.add(note[0])
        ps_seen.add(note[1])
    durations = list(durations_seen)
    approximated_sequences = []
    for sequence in new_states:
        approximated_states = []
        for state in sequence:
            if adapt_for_hmmlearn:
                state = state[0]
            if state not in original_states_flat:
                cur_dur, cur_ps = events_list[state]
                # Approximate duration, if needed
                if cur_dur not in durations_seen:
                    cur_dur = get_nearest_value(cur_dur, durations)
                # With that duration, look for the nearest value of pitch space
                delta = 1.0
                found = False
                while not found:
                    tentative_ps = max(min(cur_ps + delta, constants.MAX_PS),
                                       constants.MIN_PS)
                    if events_codes[(cur_dur, tentative_ps)] \
                            in original_states_flat:
                        found = True
                        cur_ps = tentative_ps
                    else:
                        if delta > 0:
                            delta = -delta
                        else:
                            delta = -delta + 1
                approximated_states.append(events_codes[(cur_dur, cur_ps)])
            else:
                approximated_states.append(state)
        approximated_sequences.append(approximated_states)
    return approximated_sequences


def get_states_from_songs(events_codes, songs, adapt_for_hmmlearn=False):
    obs_states = [np.array([get_repr(events_codes, s) for s in song])
                  for song in songs]
    if adapt_for_hmmlearn:
        obs_states = [state.reshape(-1, 1) for state in obs_states]
    return obs_states


def get_song_from_states(events_list, states):
    song = []
    for state in states:
        duration, ps = events_list[state]
        note = {"duration": duration, "ps": ps, "keySignature": 0,
                "timeSignature": "4/4", "restBefore": 0.0, "fermata": False}
        song.append(note)
    return song


def hmmlearn_do_test_against(songs, events_codes, events_list, model,
                             training_states, approximate=False):
    test_states = get_states_from_songs(events_codes, songs,
                                        adapt_for_hmmlearn=True)
    if approximate:
        test_states = approximate_states(test_states, training_states,
                                         events_list, events_codes,
                                         adapt_for_hmmlearn=True)
        test_states = [np.array(state).reshape(-1, 1) for state in test_states]

    likelihoods = [model.score(sequence) for sequence in
                   test_states]
    inf_number = sum(1 if math.isinf(ll) else 0 for ll in likelihoods)
    good_likelihoods = [ll for ll in likelihoods if not math.isinf(ll)]

    print("Number of inf likelihoods: {}".format(inf_number))
    print("AVG LL: {}".format(mean(good_likelihoods)))


def fhmm_do_test_against(songs, events_codes, events_list, model,
                             training_states, approximate=False):
    test_states = get_states_from_songs(events_codes, songs)

    if approximate:
        test_states = approximate_states(test_states, training_states,
                                         events_list, events_codes)
        test_states = [np.array(state) for state in test_states]

    likelihoods = [model.Forward(sequence)[2] for sequence in
                   test_states]
    inf_number = sum(1 if math.isinf(ll) or math.isnan(ll) else 0 \
                     for ll in likelihoods)
    good_likelihoods = [ll for ll in likelihoods if not math.isinf(ll)
                        and not math.isnan(ll)]

    print("Number of inf likelihoods: {}".format(inf_number))
    print("AVG LL: {}".format(mean(good_likelihoods)))


def main(skip_hmmlearn, skip_fhmm, do_generation, K, M, max_iter,
         training_size):
    songs, events_codes, events_list = init()

    bach_training = songs["bach"][:training_size]
    bach_test = songs["bach"][training_size:]

    ####### HMMLEARN ######################
    if not skip_hmmlearn:
        hmmlearn_model = hmm.MultinomialHMM(n_components=K, n_iter=max_iter)
        hmmlearn_model.monitor_.verbose = True
        hmmlearn_model.n_features = len(events_codes)
        training_states = get_states_from_songs(events_codes, bach_training,
                                                adapt_for_hmmlearn=True)
        training_lengths = [len(seq) for seq in training_states]
        # Train the model.
        hmmlearn_model.fit(np.concatenate(training_states), training_lengths)
        if do_generation:
            # Generate a new song sampling from the model.
            sampled_states, _ = \
                hmmlearn_model.sample(constants.GENERATED_SONG_SIZE)
            sampled_song = get_song_from_states(events_list, sampled_states[:, 0])
            show_sheets(sampled_song)

        # Test against Bach' test songs.
        print("Testing Bach...")
        hmmlearn_do_test_against(bach_test, events_codes, events_list,
                                 hmmlearn_model, training_states,
                                 approximate=True)

        # Test against other artists.
        print("Testing Mozart...")
        hmmlearn_do_test_against(songs["mozart"], events_codes, events_list,
                                 hmmlearn_model, training_states,
                                 approximate=True)
        print("Testing Beethoven...")
        hmmlearn_do_test_against(songs["beethoven"], events_codes, events_list,
                                 hmmlearn_model, training_states,
                                 approximate=True)
        print("Testing Einaudi...")
        hmmlearn_do_test_against(songs["einaudi"], events_codes, events_list,
                                 hmmlearn_model, training_states,
                                 approximate=True)


    ####### FACTORIAL HMM ######################
    if not skip_fhmm:
        # Init a random FHMM.
        fhmm = init_random_fhmm(constants.MAX_STEPS, len(events_codes), K, M)

        # Build the list of observable states (using appropriate codes
        # representation).
        training_states = get_states_from_songs(events_codes, bach_training)

        # Train the model.
        trained_fhmm = fhmm.EM(training_states, n_iterations=max_iter,
                               verbose=True)

        if do_generation:
            # Generate a new song sampling from the model.
            _, sampled_states = trained_fhmm.Simulate()
            sampled_song = get_song_from_states(events_list, sampled_states[0])
            show_sheets(sampled_song)

        # Test against Bach' test songs.
        print("Testing Bach...")
        fhmm_do_test_against(bach_test, events_codes, events_list,
                             trained_fhmm, training_states, approximate=True)

        # Test against other artists.
        print("Testing Mozart...")
        fhmm_do_test_against(songs["mozart"], events_codes, events_list,
                             trained_fhmm, training_states,
                             approximate=True)
        print("Testing Beethoven...")
        fhmm_do_test_against(songs["beethoven"], events_codes, events_list,
                             trained_fhmm, training_states,
                             approximate=True)
        print("Testing Einaudi...")
        fhmm_do_test_against(songs["einaudi"], events_codes, events_list,
                             trained_fhmm, training_states,
                             approximate=True)


if __name__ == "__main__":

    # Monkey-patch hmmlearn to allow non-contiguous symbols in MultinomialHMM.
    hmm.MultinomialHMM._check_input_symbols = lambda *_: True

    parser = argparse.ArgumentParser(description='HMM / FHMM on Bach music.')
    parser.add_argument('--skip-hmmlearn', dest='skip_hmmlearn',
                        action='store_true')
    parser.add_argument('--skip-fhmm', dest='skip_fhmm',
                        action='store_true')
    parser.add_argument('--do-generation', dest='do_generation',
                        action='store_true')
    parser.add_argument('-K', dest='K', action='store', type=int, default=3,
                        help="Size of hidden state alphabet.")
    parser.add_argument('-M', dest='M', action='store', type=int, default=1,
                        help="Number of markov chains (for FHMM).")
    parser.add_argument('-max-iter', dest='max_iter', action='store',
                        type=int, default=50,
                        help="Maximum number of iterations during training.")
    parser.add_argument('-training-size', dest='training_size', action='store',
                        type=int, default=30,
                        help="Number of songs (absolute value) to use in the "
                             "training set, the remaining ones will be "
                             "included in the test set")

    parser.set_defaults(skip_hmmlearn=False)
    parser.set_defaults(skip_fhmm=False)
    parser.set_defaults(do_generation=False)

    args = parser.parse_args()
    main(args.skip_hmmlearn, args.skip_fhmm, args.do_generation, args.K,
         args.M, args.max_iter, args.training_size)
