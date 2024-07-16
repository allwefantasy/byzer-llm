from typing import List, Dict, Any, Union, Optional
import pydantic


class Tag(pydantic.BaseModel):
    start_tag: str
    end_tag: str
    content: Union[str, List["Tag"], "Tag"]
    parent: Optional["Tag"] = None


class TagExtractor:
    def __init__(self, text: str):
        self.text = text
        self.pos = -1
        self.len = len(text)
        self.root_tag = Tag(start_tag="<_ROOT_>", end_tag="</_ROOT_>", content=[])
        self.current_tag = None

    def peek(self) -> str:
        if self.pos + 1 < self.len:
            return self.text[self.pos + 1]
        return ""

    def peek2(self) -> str:
        if self.pos + 1 < self.len:
            return self.text[self.pos + 2]
        return ""

    def peek3(self) -> str:
        if self.pos + 1 < self.len:
            return self.text[self.pos + 3]
        return ""

    def next(self) -> str:
        if self.pos < self.len - 1:
            self.pos += 1
            char = self.text[self.pos]
            return char
        return ""

    def is_full_tag(self) -> bool:
        return self.current_tag.start_tag and self.current_tag.end_tag

    def is_start_tag(self) -> bool:
        return self.peek() and self.peek() == "<" and self.peek2() == "_"

    def extract_start_tag(self) -> str:
        tag = ""

        while self.peek() and self.peek() != ">":
            tag += self.next()
        tag += self.next()

        if self.current_tag is None or self.current_tag == self.root_tag:
            self.current_tag = Tag(
                start_tag=tag, end_tag="", content="", parent=self.root_tag
            )
            self.root_tag.content.append(self.current_tag)
        ## 当前tag已经闭合，找到当前tag的父tag,遇到新tag
        elif self.is_full_tag():
            parent = self.current_tag.parent or self.current_tag
            current_tag = Tag(
                start_tag=tag,
                end_tag="",
                content="",
                parent=parent,
            )
            if isinstance(parent.content, list):
                parent.content.append(current_tag)
            else:
                s = self.current_tag.content
                parent.content = []
                if s:
                    parent.content.append(s)
                parent.content.append(current_tag)
            self.current_tag = current_tag
        ## 当前tag还没有闭合，直接添加到当前tag的content中
        elif not self.is_full_tag():
            current_tag = Tag(
                start_tag=tag, end_tag="", content="", parent=self.current_tag
            )
            if isinstance(self.current_tag.content, list):
                self.current_tag.content.append(current_tag)
            else:
                s = self.current_tag.content
                self.current_tag.content = []
                if s:
                    self.current_tag.content.append(s)
                self.current_tag.content.append(current_tag)
            self.current_tag = current_tag

        return tag

    def is_end_tag(self) -> bool:
        return (
            self.peek()
            and self.peek() == "<"
            and self.peek2() == "/"
            and self.peek3() == "_"
        )

    def extract_end_tag(self) -> str:
        tag = ""
        while self.peek() and self.peek() != ">":
            tag += self.next()
        tag += self.next()
        self.current_tag.end_tag = tag
        self.current_tag = self.current_tag.parent
        return tag

    def consume_blank(self):
        while (
            self.peek() == "\n"
            or self.peek() == " "
            or self.peek() == "\t"
            or self.peek() == "\r"
        ):
            self.next()

    def is_in_tag(self) -> bool:
        return (
            self.current_tag
            and self.current_tag.start_tag
            and not self.current_tag.end_tag
        )

    def is_tag_content(self) -> bool:
        if not self.root_tag:
            return False
        temp_pos = self.pos
        self.consume_blank()
        if self.is_start_tag():
            self.pos = temp_pos            
            return True
        if self.is_end_tag():
            self.pos = temp_pos
            return True                
        return False

    def extract_str_content(self) -> str:
        content = ""
        while not self.is_start_tag() and not self.is_end_tag():
            content += self.next()
         
        self.current_tag.content = content
        return content

    def is_not_in_tag_str(self) -> bool:
        if not self.root_tag:
            return True
        if not self.current_tag:
            return True
        if not self.current_tag.start_tag and not self.current_tag.end_tag:
            return True
        if self.current_tag.start_tag and self.current_tag.end_tag:
            return True
        return False

    def extract_content_not_in_tag(self) -> str:
        content = ""
        while self.peek() and not self.is_start_tag() and not self.is_end_tag():
            content += self.next()
        return content

    def extract(self) -> Union[Tag]:
        while True:
            if self.pos == self.len - 1:
                break
            if self.is_start_tag():
                self.extract_start_tag()
            elif self.is_end_tag():
                self.extract_end_tag()
            elif self.is_in_tag():
                if self.is_tag_content():
                    self.consume_blank()
                    continue
                else:
                    self.extract_str_content()
            elif self.is_not_in_tag_str():
                self.extract_content_not_in_tag()            
        return self.root_tag
