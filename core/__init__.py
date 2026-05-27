import matplotlib.pyplot as plt
from matplotlib import font_manager

_KOREAN_FONT_CANDIDATES = [
    "NanumGothic",
    "NanumBarunGothic",
    "NanumSquareRound",
    "Noto Sans CJK KR",
    "Malgun Gothic",
]


def set_korean_font() -> None:
    """시스템에서 사용 가능한 한글 폰트를 matplotlib 기본 폰트로 설정한다."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _KOREAN_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    raise RuntimeError(f"한글 폰트를 찾을 수 없습니다. 후보: {_KOREAN_FONT_CANDIDATES}")
