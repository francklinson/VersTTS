#!/usr/bin/env python3
"""
录音参考文本路由
提供供用户朗读录音的参考文本片段
"""

from typing import Optional, List, Dict
from fastapi import APIRouter, Query, HTTPException
import random

from backend.logger_config import system_logger

router = APIRouter()


# 预定义的朗读文本片段
RECORDING_SCRIPTS = {
    "short": [
        # 经典诗词 - 唐诗
        {
            "id": "poem_001",
            "text": "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
            "type": "唐诗",
            "source": "李白《静夜思》",
            "duration": "6-9秒"
        },
        {
            "id": "poem_002",
            "text": "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
            "type": "唐诗",
            "source": "孟浩然《春晓》",
            "duration": "7-10秒"
        },
        {
            "id": "poem_003",
            "text": "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
            "type": "唐诗",
            "source": "王之涣《登鹳雀楼》",
            "duration": "8-11秒"
        },
        # 经典诗词 - 宋词
        {
            "id": "poem_101",
            "text": "明月几时有？把酒问青天。不知天上宫阙，今夕是何年。",
            "type": "宋词",
            "source": "苏轼《水调歌头》",
            "duration": "9-12秒"
        },
        # 现代散文
        {
            "id": "prose_001",
            "text": "盼望着，盼望着，东风来了，春天的脚步近了。一切都像刚睡醒的样子，欣欣然张开了眼。",
            "type": "现代散文",
            "source": "朱自清《春》",
            "duration": "10-13秒"
        },
        # 新闻播报
        {
            "id": "news_001",
            "text": "今日上午，我国新一代载人飞船试验船在文昌航天发射场成功发射，标志着我国载人航天工程进入新的发展阶段。",
            "type": "新闻",
            "source": "综合新闻",
            "duration": "11-14秒"
        },
        # 日常对话
        {
            "id": "dialogue_001",
            "text": "你好，请问附近有什么好吃的餐厅吗？我想尝尝当地的特色美食。",
            "type": "对话",
            "source": "日常用语",
            "duration": "6-9秒"
        },
        {
            "id": "dialogue_002",
            "text": "今天的天气真不错，阳光明媚，适合出去走走。周末有什么计划吗？",
            "type": "对话",
            "source": "日常用语",
            "duration": "8-11秒"
        },
        # 故事叙述
        {
            "id": "story_001",
            "text": "从前有座山，山上有座庙，庙里有个老和尚和一个小和尚。有一天，小和尚问老和尚...",
            "type": "故事",
            "source": "民间故事",
            "duration": "9-12秒"
        }
    ],
    "medium": [
        # 中等长度文本
        {
            "id": "medium_001",
            "text": "人工智能正在深刻改变我们的生活方式。从智能手机到自动驾驶，从语音助手到智能医疗，AI技术已经渗透到社会的方方面面。它不仅提高了生产效率，也为人们带来了更加便捷的生活体验。",
            "type": "科普",
            "source": "科技短文",
            "duration": "12-16秒"
        },
        {
            "id": "medium_002",
            "text": "读书是一种美好的享受。在书的世界里，我们可以穿越时空，与古今中外的智者对话；可以开阔视野，了解不同文化背景下的思想精华；可以净化心灵，在喧嚣的世界中找到一片宁静。",
            "type": "叙述",
            "source": "阅读随笔",
            "duration": "13-17秒"
        },
        {
            "id": "medium_003",
            "text": "健康饮食对于维持良好的身体状态至关重要。我们应该多吃蔬菜水果，适量摄入蛋白质，减少油腻和高糖食物的摄入。同时，规律的饮食习惯和充足的水分补充也是保持健康的关键。",
            "type": "科普",
            "source": "健康指南",
            "duration": "12-16秒"
        }
    ],
    "long": [
        # 较长文本
        {
            "id": "long_001",
            "text": "尊敬的各位来宾，大家好！今天我非常荣幸能够站在这里，与大家分享我的一些想法。在这个充满机遇与挑战的时代，我们每个人都肩负着不同的责任和使命。无论是科技创新、文化传承，还是社会服务，都需要我们付出努力和汗水。让我们携手并进，共同创造更加美好的未来！",
            "type": "演讲",
            "source": "演讲稿",
            "duration": "18-23秒"
        },
        {
            "id": "long_002",
            "text": "那座古老的城堡坐落在山顶之上，周围环绕着茂密的森林。传说中，这里曾经是一位勇敢骑士的家园，他用自己的智慧和勇气保护了这片土地数百年。如今，虽然城堡已经年久失修，但那些关于勇气与荣誉的故事依然在村民口中代代相传，激励着一代又一代的年轻人。",
            "type": "故事",
            "source": "历史传说",
            "duration": "19-24秒"
        },
        # 诗歌长诗
        {
            "id": "poem_201",
            "text": "轻轻的我走了，正如我轻轻的来；我轻轻的招手，作别西天的云彩。那河畔的金柳，是夕阳中的新娘；波光里的艳影，在我的心头荡漾。软泥上的青荇，油油的在水底招摇；在康河的柔波里，我甘心做一条水草！那榆荫下的一潭，不是清泉，是天上虹；揉碎在浮藻间，沉淀着彩虹似的梦。",
            "type": "现代诗",
            "source": "徐志摩《再别康桥》",
            "duration": "23-29秒"
        }
    ]
}


@router.get("/")
async def get_recording_scripts(
        length: str = Query("short", description="文本长度: short/medium/long"),
        type_filter: Optional[str] = Query(None, description="文本类型过滤")
):
    """
    获取供用户朗读录音的参考文本片段
    
    参数:
    - length: 文本长度 (short-短文本5-8秒, medium-中等10-15秒, long-长文本15-20秒)
    - type_filter: 文本类型过滤 (通用/描述/新闻/对话/叙述/科普/演讲/故事)
    """
    try:
        scripts = RECORDING_SCRIPTS.get(length, RECORDING_SCRIPTS["short"])

        # 应用类型过滤
        if type_filter:
            scripts = [s for s in scripts if s.get("type") == type_filter]

        return {
            "success": True,
            "length": length,
            "count": len(scripts),
            "scripts": scripts
        }
    except Exception as e:
        system_logger.error(f"获取录音文本失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取录音文本失败: {str(e)}")


@router.get("/types")
async def get_recording_script_types():
    """获取所有可用的录音文本类型"""
    types = set()
    for scripts in RECORDING_SCRIPTS.values():
        for script in scripts:
            types.add(script.get("type", "通用"))

    return {
        "success": True,
        "types": sorted(list(types))
    }


@router.get("/random")
async def get_random_recording_script(
        length: str = Query("short", description="文本长度: short/medium/long"),
        type_filter: Optional[str] = Query(None, description="文本类型过滤")
):
    """
    随机获取一条录音参考文本
    
    参数:
    - length: 文本长度
    - type_filter: 文本类型过滤
    """
    try:
        scripts = RECORDING_SCRIPTS.get(length, RECORDING_SCRIPTS["short"])

        # 应用类型过滤
        if type_filter:
            scripts = [s for s in scripts if s.get("type") == type_filter]

        if not scripts:
            raise HTTPException(status_code=404, detail="未找到符合条件的文本")

        # 随机选择一条
        script = random.choice(scripts)

        return {
            "success": True,
            "script": script
        }
    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(f"获取随机录音文本失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取随机录音文本失败: {str(e)}")
