import exmel
import cProfile

def main():
    exmel.align(
        "dataset/不为谁而作的歌/不为谁而作的歌.m.mid",
        "dataset/不为谁而作的歌/不为谁而作的歌.t.mid")

cProfile.run("main()", sort='cumtime')