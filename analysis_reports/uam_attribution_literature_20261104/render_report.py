"""仅机械渲染本次文献报告，不导入模型库或执行实验。"""

import json
from pathlib import Path
import runpy


REPORT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_ROOT.parent.parent


def main():
    # 沿用已有排版函数，不运行旧报告脚本的主入口。
    renderer = runpy.run_path(
        str(REPORT_ROOT.parent / "uam_three_batch_review_20261104" / "render_report.py")
    )["render"]
    source = PROJECT_ROOT / "allinone.md"
    provenance = json.loads((REPORT_ROOT / "provenance.json").read_text(encoding="utf-8"))
    page = renderer(source.read_text(encoding="utf-8"))
    page = page.replace(
        "<title>三组实验合并分析与后续安排</title>",
        "<title>强化学习归因文献与UAM诊断改进</title>",
    ).replace(
        "数据统计复算完成；未启动新训练或服务器任务。正文第一节是简明结论。",
        "10篇代表性论文查核；方案待实施；没有新增训练、仿真或服务器任务。先读第1节和第4节。",
    ).replace(
        'href="analysis_reports/uam_three_batch_review_20261104/',
        'href="../uam_three_batch_review_20261104/',
    )
    # 确认来源记录与正文的结构完整，避免把渲染成功误报成实验验证。
    assert len(provenance["sources"]) == 10
    assert page.count("<table>") == 7
    target = REPORT_ROOT / "01_强化学习归因文献与UAM诊断改进.html"
    target.write_text(page, encoding="utf-8")
    print(f"核验来源：{len(provenance['sources'])}篇")
    print(f"正文字符数：{len(source.read_text(encoding='utf-8'))}")
    print(f"阅读版：{target}")


if __name__ == "__main__":
    main()
