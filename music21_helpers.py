import music21
import pickle
import os


def parse_song(score, chosenPart=None):
    song = []
    # Get only the part with chosenPart sheet
    if chosenPart is not None:
        try:
            part = score.parts[chosenPart]
        except:
            part = None
        # if there isn't a part return None
        if part is None:
            return None
    else:
        part = score.parts[0]

    # To see part structure with offsets, keySignature, timeSignature, ...
    # part.flat.show('text')

    # keySig and timeSig are objects, not single properties. Stringify them?
    last_keySig = part.flat.keySignature.sharps
    last_timeSig = part.flat.timeSignature.ratioString
    rest = 0.0

    for element in part.flat.elements:
        if isinstance(element, music21.note.Note):
            note = dict()
            note["name"] = element.pitch.name
            note["nameWithOctave"] = element.nameWithOctave
            note["octave"] = element.pitch.octave
            note["duration"] = element.duration.quarterLength
            note["ps"] = element.pitch.ps
            note["keySignature"] = last_keySig
            note["timeSignature"] = last_timeSig
            note["restBefore"] = rest
            note["fermata"] = len(element.expressions) > 0 and isinstance(element.expressions[0],
                                                                          music21.expressions.Fermata)
            rest = 0.0
            song.append(note)

        elif isinstance(element, music21.note.Rest):
            rest += element.duration.quarterLength

        elif isinstance(element, music21.key.KeySignature):
            last_keySig = element.sharps

        elif isinstance(element, music21.meter.TimeSignature):
            last_timeSig = element.ratioString

    if len(song) == 0:
        breakpoint_here = 1
    return song


def create_dataset_from_corpus():
    i = 1
    artists = ["bach", "beethoven", "mozart"]
    instruments = ["Soprano", "Viola", "Viola"]

    # Create a dataset for every artist and save them using pickle
    for j in range(len(artists)):
        artist = artists[j]
        instrument = instruments[j]
        scores = music21.corpus.search(artist, "composer")
        songs = []
        for score in scores:
            song = parse_song(score.parse(), chosenPart=instrument)

            # Do not add to dataset empty or missing part songs
            if song is None or len(song) == 0:
                continue

            print(i, instrument)
            i += 1
            songs.append(song)

        # dump dataset to file
        with open("dataset/%s.pickle" % artist, "wb") as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(songs, f, pickle.HIGHEST_PROTOCOL)
            print("Wrote all %s songs to file. Next one!" % artist)
            i = 1


def create_dataset_from_external_musicxml(folder=None):
    filenames = ["Bohemian_Rhapsody_for_Piano.musicxml"]
    if folder:
        filenames = []
        for file in os.listdir("dataset/%s" % folder):
            if file.endswith(".musicxml"):
                filenames.append(os.path.join(folder, file))
    songs = []
    i = 1
    for filename in filenames:
        importer = music21.musicxml.xmlToM21.MusicXMLImporter()
        score = importer.scoreFromFile("dataset/%s" % filename)
        songs.append(parse_song(score))
        print(i)
        i += 1

    with open("dataset/%s.pickle" % folder, "wb") as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(songs, f, pickle.HIGHEST_PROTOCOL)
        print("Wrote %s.pickle" % folder)


def show_sheets(song):
    # enable musescore to view notes sheet
    music21.environment.set('musicxmlPath', '/usr/bin/musescore')

    # Generate music from lists of notes
    stream = music21.stream.Stream()
    stream.keySignature = music21.key.KeySignature(song[0]["keySignature"])
    stream.timeSignature = music21.meter.TimeSignature(song[0]["timeSignature"])
    for note in song:
        stream.append(music21.note.Note(ps=note["ps"], quarterLength=note["duration"]))

    stream.show()

# generate MIDI from stream
# mf = music21.midi.translate.streamToMidiFile(stream)
# mf.open("midi.mid", "wb")
# mf.write()
# mf.close()
#
# music21.environment.set('musicxmlPath', '/usr/bin/musescore')
# chorale = music21.corpus.parse('bach/bwv324.xml')
# chorale.show()
#
# create_dataset_from_external_musicxml("einaudi")
# create_dataset_from_corpus()