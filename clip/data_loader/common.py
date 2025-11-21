import html
import ftfy
import regex as re


def text_clean(text):    

    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r'\s+', ' ', text).strip().lower()
    

def start_tag_sentence():

    return "<|satrt_of_sentence|>"


def end_tag_sentence():

    return "<|end_of_sentence|>"


def end_tag_word():

    return "<|end_of_word|>"


def bytes_to_unicode():

    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]

    return dict(zip(bs, cs))