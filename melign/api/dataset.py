from pathlib import Path
from typing import Any, Iterable, Iterator, TypedDict, overload
from dataclasses import dataclass
from tinytag import TinyTag

from melign.data.io import PathLike
from melign.data.sequence import Melody, Performance

@dataclass(frozen=True, slots=True)
class Song:
    name: str
    melody: Melody
    performance: Performance
    performance_path: Path
    ground_truth: Melody | None
    baseline: Melody | None
    audio_path: Path | None

class DatasetConfig(TypedDict):
    melody_ext: str | Iterable[str]
    performance_ext: str | Iterable[str]
    ground_truth_ext: str | Iterable[str]
    ground_truth_track_idx: int | None
    baseline_ext: str | Iterable[str]
    baseline_track_idx: int | None
    audio_ext: str | Iterable[str]

default_config: DatasetConfig = {
    "melody_ext": (".m.mid", ".note", ".est.note"),
    "performance_ext": ".t.mid",
    "ground_truth_ext": ".gt.mid",
    "ground_truth_track_idx": 3,
    "baseline_ext": ".bl.mid",
    "baseline_track_idx": 3,
    "audio_ext": (".mp3", ".opus"),
}

class Dataset:
    """
    A dataset of songs, organized in this structure:
    
    dataset/
    ├── song one/
    │   ├── song one.m.mid (required: reference melody)
    │   ├── song one.t.mid (required: piano transcription)
    │   ├── song one.gt.mid (optional: ground truth)
    │   ├── song one.bl.mid (optional: baseline)
    │   └── song one.mp3 (optional: audio)
    ├── song two/
    │   └── ... (files for <song two>)
    └── ... (other song directories)
    """
    def __init__(
        self,
        source: 'DatasetLike',
        config: DatasetConfig = default_config,
        truncate_performance: bool = True,
        enable_cache: bool = True,
    ):
        self.config = config
        self.truncate_performance = truncate_performance
        self.enable_cache = enable_cache
        self._cache: dict[str, Song] = {}
        
        if isinstance(source, Dataset):
            self.song_dirs = [d for d in source.song_dirs]
        elif isinstance(source, (Path, str)):
            self.song_dirs = [d for d in Path(source).iterdir() if d.is_dir()]
        elif isinstance(source, Iterable):
            self.song_dirs = [Path(d) for d in source]
        if len(self.song_dirs) == 0:
            raise FileNotFoundError(f"No directories found in {source}")

    def _get_file_path(self, song_dir: Path, song_name: str, ext: str | Iterable[str]) -> Path | None:
        if isinstance(ext, str):
            ext = [ext]
        for e in ext:
            if (song_dir / f"{song_name}{e}").exists():
                return song_dir / f"{song_name}{e}"
        return None

    def __len__(self):
        return len(self.song_dirs)

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about the cache status."""
        return {
            "cache_enabled": self.enable_cache,
            "cache_size": len(self._cache),
            "total_songs": len(self.song_dirs),
            "cached_songs": list(self._cache.keys())
        }

    @overload
    def __getitem__(self, query: int | str) -> Song: ...
    @overload
    def __getitem__(self, query: slice | Iterable[str]) -> 'Dataset': ...
    def __getitem__(self, query: int | str | slice | Iterable[str]) -> 'Song | Dataset':
        # Handle string queries by finding the song with that name
        if isinstance(query, str):
            for i, song_dir in enumerate(self.song_dirs):
                if song_dir.name == query:
                    return self[i]
            else:
                raise KeyError(f"Song '{query}' not found in dataset")
        if isinstance(query, slice):
            return Dataset(self.song_dirs[query])
        if isinstance(query, Iterable):
            return Dataset([dir_ for dir_ in self.song_dirs if dir_.name in query])

        song_dir = self.song_dirs[query]
        song_name = song_dir.name

        # Check if the song is in the cache
        if self.enable_cache and song_name in self._cache:
            return self._cache[song_name]
        
        # Required files
        melody_path = self._get_file_path(song_dir, song_name, self.config["melody_ext"])
        performance_path = self._get_file_path(song_dir, song_name, self.config["performance_ext"])
        
        if melody_path is None:
            raise FileNotFoundError(f"Required melody file not found: {melody_path}")
        if performance_path is None:
            raise FileNotFoundError(f"Required performance file not found: {performance_path}")
        
        melody = Melody(melody_path)
        performance = Performance(performance_path)
        
        # Optional files - create objects only if files exist
        ground_truth_path = self._get_file_path(song_dir, song_name, self.config["ground_truth_ext"])
        if ground_truth_path is not None:
            ground_truth = Melody(ground_truth_path, track_idx=self.config["ground_truth_track_idx"])
        else:
            ground_truth = None
        
        baseline_path = self._get_file_path(song_dir, song_name, self.config["baseline_ext"])
        if baseline_path is not None:
            baseline = Melody(baseline_path, track_idx=self.config["baseline_track_idx"])
        else:
            baseline = None
        
        audio_path = self._get_file_path(song_dir, song_name, self.config["audio_ext"])
        if audio_path is not None:
            audio_path = Path(audio_path)
        else:
            audio_path = None

        if self.truncate_performance:
            assert audio_path is not None, f"Audio file is required to truncate performance for {song_name}"
            performance = self._truncate_performance(performance, audio_path)
        
        song = Song(song_name, melody, performance, performance_path, ground_truth, baseline, audio_path)
        
        # Cache the result if caching is enabled
        if self.enable_cache:
            self._cache[song_name] = song
        
        return song

    def __iter__(self) -> Iterator[Song]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        cache_info = f", cache_size={len(self._cache)}" if self.enable_cache else ""
        return f"Dataset(source={self.song_dirs.__repr__()}{cache_info})"

    def is_full(self) -> bool:
        if not self.is_valid():
            return False
        for song_dir in self.song_dirs:
            song_name = song_dir.name
            if (
                self._get_file_path(song_dir, song_name, self.config["ground_truth_ext"]) is None or
                self._get_file_path(song_dir, song_name, self.config["baseline_ext"]) is None or
                self._get_file_path(song_dir, song_name, self.config["audio_ext"]) is None
            ):
                return False
        return True

    def is_valid(self) -> bool:
        for song_dir in self.song_dirs:
            song_name = song_dir.name
            if (
                self._get_file_path(song_dir, song_name, self.config["melody_ext"]) is None or
                self._get_file_path(song_dir, song_name, self.config["performance_ext"]) is None or
                (self.truncate_performance and self._get_file_path(song_dir, song_name, self.config["audio_ext"]) is None)
            ):
                return False
        return True

    def _truncate_performance(self, performance: Performance, audio_path: Path) -> Performance:
        audio_duration = TinyTag.get(audio_path).duration
        assert audio_duration is not None, f"Audio duration is None for {audio_path}"
        midi_duration = performance.duration
        if midi_duration > audio_duration:
            # print(f"Performance Midi duration ({midi_duration:.2f}s) is longer than audio duration ({audio_duration:.2f}s) for {audio_path}")
            return performance[None, audio_duration]
        else:
            return performance
            
type DatasetLike = Dataset | PathLike | Iterable[PathLike]