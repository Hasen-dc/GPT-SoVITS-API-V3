#! /usr/bin/env python
# -*- coding: utf-8 -*-


class CutUtil(object):
    splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
    punctuation = set(['!', '?', '…', ',', '.', '-', " "])
    cut_method_names = ["cut0", "cut1", "cut2", "cut3", "cut4", "cut5"]

    def __init__(self):
        pass

    @staticmethod
    def cut_text(how_to_cut, text):
        if how_to_cut == "cut1":
            text = CutUtil.cut1(text)
        elif how_to_cut == "cut2":
            text = CutUtil.cut2(text)
        elif how_to_cut == "cut3":
            text = CutUtil.cut3(text)
        elif how_to_cut == "cut4":
            text = CutUtil.cut4(text)
        elif how_to_cut == "cut5":
            text = CutUtil.cut5(text)
        return text

    @staticmethod
    def split(todo_text):
        todo_text = todo_text.replace("……", "。").replace("——", "，")
        if todo_text[-1] not in CutUtil.splits:
            todo_text += "。"
        i_split_head = i_split_tail = 0
        len_text = len(todo_text)
        todo_texts = []
        while 1:
            if i_split_head >= len_text:
                break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
            if todo_text[i_split_head] in CutUtil.splits:
                i_split_head += 1
                todo_texts.append(todo_text[i_split_tail:i_split_head])
                i_split_tail = i_split_head
            else:
                i_split_head += 1
        return todo_texts

    # 不切
    @staticmethod
    def cut0(inp):
        if not set(inp).issubset(CutUtil.punctuation):
            return inp
        else:
            return "/n"

    # 凑四句一切
    @staticmethod
    def cut1(inp):
        inp = inp.strip("\n")
        inps = CutUtil.split(inp)
        split_idx = list(range(0, len(inps), 4))
        split_idx[-1] = None
        if len(split_idx) > 1:
            opts = []
            for idx in range(len(split_idx) - 1):
                opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
        else:
            opts = [inp]
        opts = [item for item in opts if not set(item).issubset(CutUtil.punctuation)]
        return "\n".join(opts)

    # 凑50字一切
    @staticmethod
    def cut2(inp):
        inp = inp.strip("\n")
        inps = CutUtil.split(inp)
        if len(inps) < 2:
            return inp
        opts = []
        summ = 0
        tmp_str = ""
        for i in range(len(inps)):
            summ += len(inps[i])
            tmp_str += inps[i]
            if summ > 50:
                summ = 0
                opts.append(tmp_str)
                tmp_str = ""
        if tmp_str != "":
            opts.append(tmp_str)
        # print(opts)
        if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
            opts[-2] = opts[-2] + opts[-1]
            opts = opts[:-1]
        opts = [item for item in opts if not set(item).issubset(CutUtil.punctuation)]
        return "\n".join(opts)

    # 按中文句号。切
    @staticmethod
    def cut3(inp):
        inp = inp.strip("\n")
        opts = ["%s" % item for item in inp.strip("。").split("。")]
        opts = [item for item in opts if not set(item).issubset(CutUtil.punctuation)]
        return "\n".join(opts)

    # 按英文句号.切
    @staticmethod
    def cut4(inp):
        inp = inp.strip("\n")
        opts = ["%s" % item for item in inp.strip(".").split(".")]
        opts = [item for item in opts if not set(item).issubset(CutUtil.punctuation)]
        return "\n".join(opts)

    # 按标点符号切
    # contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
    @staticmethod
    def cut5(inp):
        inp = inp.strip("\n")
        punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
        mergeitems = []
        items = []

        for i, char in enumerate(inp):
            if char in punds:
                if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                    items.append(char)
                else:
                    items.append(char)
                    mergeitems.append("".join(items))
                    items = []
            else:
                items.append(char)

        if items:
            mergeitems.append("".join(items))

        opt = [item for item in mergeitems if not set(item).issubset(punds)]
        return "\n".join(opt)
