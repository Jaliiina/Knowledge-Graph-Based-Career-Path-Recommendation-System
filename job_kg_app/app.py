import os
import csv
import time
import warnings
from flask import Flask, render_template, request, jsonify, session
from flask_wtf import FlaskForm
from wtforms import HiddenField, validators
from neo4j import GraphDatabase
from openai import OpenAI

# 忽略无关警告
warnings.filterwarnings("ignore", category=UserWarning, module='jieba')

# ========== 初始化配置 ==========
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['JSON_AS_ASCII'] = False  # 解决中文乱码
app.config['SESSION_TYPE'] = 'filesystem'

# Neo4j配置
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# DeepSeek API配置
DEEPSEEK_API_KEY = ""
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# ========== 技能词典加载 ==========
SKILL_CSV_PATH = "job_kg_app/skill_nodes.csv"


def load_skill_dict():
    """加载技能词典（去重、排序）"""
    skill_set = set()
    try:
        with open(SKILL_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                if row and row[0].strip():
                    skill_set.add(row[0].strip())
    except FileNotFoundError:
        # 内置默认技能（无CSV时兜底）
        skill_set = {"Java", "Python", "MySQL", "Redis", "Spring Boot", "Vue.js", "React.js", "JavaScript", "HTML",
                     "CSS"}
    return sorted(list(skill_set))


# 全局技能列表
SKILL_LIST = load_skill_dict()


# ========== 表单定义 ==========
class SkillForm(FlaskForm):
    skills = HiddenField('技能列表', validators=[validators.DataRequired(message="请至少添加一个技能")])


# ========== 工具函数 ==========
def filter_none_skills(skills_list):
    """过滤掉技能列表中的None值和空字符串"""
    if not skills_list:
        return []
    # 确保所有元素转换为字符串并去除空白
    filtered = []
    for skill in skills_list:
        if skill is None:
            continue
        # 转换为字符串并去空格
        skill_str = str(skill).strip()
        if skill_str:  # 非空字符串
            filtered.append(skill_str)
    return filtered


def safe_join(skills_list, separator='、'):
    """安全连接技能列表，过滤None值"""
    filtered = filter_none_skills(skills_list)
    return separator.join(filtered)


def get_safe_slice(skills_list, count):
    """安全获取技能列表切片"""
    filtered = filter_none_skills(skills_list)
    return filtered[:min(count, len(filtered))]


def get_match_level(score: int) -> str:
    """获取匹配等级"""
    if score >= 80:
        return "优秀匹配"
    elif score >= 60:
        return "良好匹配"
    elif score >= 40:
        return "一般匹配"
    else:
        return "待提升"


def get_competition_summary(score, owned_skills, missing_skills):
    """生成竞争力总结文本"""
    # 过滤None值
    owned_skills = filter_none_skills(owned_skills)
    missing_skills = filter_none_skills(missing_skills)

    if score >= 80:
        return f"✅ 你的技能匹配度高达{score}%，已掌握{len(owned_skills)}项核心技能，远超岗位基础要求！"
    elif score >= 60:
        return f"⚠️ 你的技能匹配度{score}%，已掌握{len(owned_skills)}项核心技能，但需补充{len(missing_skills)}项关键技能！"
    elif score >= 40:
        # 安全获取前2个技能
        top_missing = get_safe_slice(missing_skills, 2)
        missing_text = ", ".join(top_missing) if top_missing else "关键技能"
        return f"📚 你的技能匹配度{score}%，仅掌握{len(owned_skills)}项核心技能，建议优先学习{missing_text}！"
    else:
        # 安全获取前3个技能
        top_missing = get_safe_slice(missing_skills, 3)
        missing_text = ", ".join(top_missing) if top_missing else "核心技能"
        return f"🔧 你的技能匹配度仅{score}%，需系统学习{missing_text}等核心技能！"


def generate_llm_report(match_result, user_skills):
    """调用DeepSeek API生成分析报告"""
    # 过滤掉None值
    owned_skills = filter_none_skills(match_result.get('owned_skills', []))
    missing_skills = filter_none_skills(match_result.get('missing_skills', []))
    recommend_skills = filter_none_skills(match_result.get('recommend_skills', []))

    prompt = f"""
        请基于以下岗位匹配信息，生成一份排版清晰的智能分析报告：
        - 目标岗位：{match_result['job_name']}
        - 匹配分数：{match_result['match_score']}%
        - 已具备技能：{safe_join(owned_skills, '、') if owned_skills else '无'}
        - 缺失技能：{safe_join(missing_skills, '、') if missing_skills else '无'}
        - 优先补齐建议：{safe_join(recommend_skills, '、') if recommend_skills else '无'}

        报告必须包含以下板块：
        1. 【匹配情况总结】
        2. 【已具备技能优势】
        3. 【学习优先级建议】
        4. 【简历项目描述优化建议】（重点：基于现有技能，怎么突出与岗位的适配度）
        5. 【行动小贴士】

        格式要求：
        - 每个板块用【标题】开头；
        - 两行文字中间不需要间隔一行；
        - 必须分行，不能堆积成一段！！！
        - 语言口语化，避免大段文字；
        - 内容用短句，每段不超过2行，关键信息用"✅""⚠️"标记；
        - 不要用任何Markdown格式，只输出纯文本+换行符。
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位职业技能分析助手"},
                {"role": "user", "content": prompt.strip()}
            ],
            stream=False,
            timeout=30
        )
        # 关键：将纯文本的换行符替换为HTML的<br>标签，确保网页渲染换行
        report_html = response.choices[0].message.content.replace("\n", "<br>")
        return report_html
        # return response.choices[0].message.content
    except Exception as e:
        return f"智能报告生成失败：{str(e)}"


def create_user_skill_relation(user_id, skills):
    """创建用户-技能关联（覆盖旧关系）"""
    try:
        # 过滤掉None值
        skills = filter_none_skills(skills)
        if not skills:
            return False

        with neo4j_driver.session() as session:
            # 创建用户节点（存在则更新）
            session.run("MERGE (u:Person {id: $user_id}) SET u.name = '手动输入用户'", user_id=user_id)
            # 删除旧技能关系
            session.run("MATCH (u:Person {id: $user_id})-[r:HAS_SKILL]->() DELETE r", user_id=user_id)
            # 批量创建新技能关系
            for skill in skills:
                session.run("""
                    MERGE (s:Skill {name: $skill})
                    MERGE (u:Person {id: $user_id})-[r:HAS_SKILL]->(s)
                """, skill=skill, user_id=user_id)
        return True
    except Exception as e:
        print(f"[ERROR] 创建技能关系失败：{e}")
        return False


# ========== 核心路由 ==========
@app.route('/')
def home():
    """首页"""
    return render_template('home.html')


@app.route('/api/skill/suggest', methods=['GET'])
def skill_suggest():
    """技能联想接口"""
    prefix = request.args.get('prefix', '').strip().lower()
    if not prefix:
        return jsonify({"code": 0, "data": []})

    match_skills = [
        skill for skill in SKILL_LIST
        if skill.lower().startswith(prefix)
    ]
    return jsonify({"code": 0, "data": match_skills[:10]})


# ========== 功能路由 ==========
@app.route("/explore")
def explore_page():
    """
    职位图谱浏览：
    - 支持关键词搜索（职位名 / 城市，模糊匹配）
    - 左边岗位列表
    - 右边选中岗位的技能子图信息 + 前端画图所需数据
    """
    q = (request.args.get("q") or "").strip()
    selected_job_id = (request.args.get("job_id") or "").strip() or None

    stats = {}
    jobs = []
    selected_job = None
    job_skills = []

    try:
        with neo4j_driver.session() as session:
            # 统计节点 / 关系总数
            stats_rec = session.run("""
                CALL {
                  MATCH (j:Job) RETURN count(j) AS job_count
                }
                CALL {
                  MATCH (s:Skill) RETURN count(s) AS skill_count
                }
                CALL {
                  MATCH ()-[r:REQUIRES]->() RETURN count(r) AS rel_count
                }
                RETURN job_count, skill_count, rel_count
            """).single()

            if stats_rec:
                stats = {
                    "job_count": stats_rec["job_count"],
                    "skill_count": stats_rec["skill_count"],
                    "rel_count": stats_rec["rel_count"],
                }

            # 带模糊搜索的岗位列表
            jobs_query = """
                MATCH (j:Job)
                WHERE $q = "" 
                   OR toLower(coalesce(j.name, j.title, "")) CONTAINS toLower($q)
                   OR toLower(coalesce(j.city, "")) CONTAINS toLower($q)
                RETURN 
                  j.job_id AS id,
                  coalesce(j.name, j.title, "") AS name,
                  coalesce(j.city, "未知") AS city
                ORDER BY id
                LIMIT 200
            """
            jobs = session.run(jobs_query, q=q).data()

            # 默认选第一个
            if jobs and not selected_job_id:
                selected_job_id = jobs[0]["id"]

            if selected_job_id:
                rec = session.run("""
                    MATCH (j:Job {job_id: $job_id})
                    OPTIONAL MATCH (j)-[r:REQUIRES]->(s:Skill)
                    RETURN 
                      j.job_id AS id,
                      coalesce(j.name, j.title, "") AS name,
                      coalesce(j.city, "未知") AS city,
                      collect({skill: s.name, weight: coalesce(r.weight, 1.0)}) AS skills
                """, job_id=selected_job_id).single()

                if rec:
                    selected_job = {
                        "id": rec["id"],
                        "name": rec["name"],
                        "city": rec["city"],
                    }
                    # 过滤掉技能为None的条目
                    job_skills = [x for x in rec["skills"] if x["skill"] is not None and x["skill"].strip() != ""]

    except Exception as e:
        print("[ERROR] explore_page:", e)

    # 把搜索词也传给模板
    return render_template(
        "explore.html",
        q=q,
        jobs=jobs,
        stats=stats,
        selected_job=selected_job,
        job_skills=job_skills,
    )


@app.route("/resume-kg", methods=["GET", "POST"])
def resume_kg_page():
    """简历技能解析：从简历文本中抽技能，并写入知识图谱"""
    user_id = "manual_input_user"  # 和 match_diag / path_reco 共用一个用户
    resume_text = ""
    extracted_skills = []
    message = None
    graph_skills = []

    # 处理表单提交
    if request.method == "POST":
        resume_text = request.form.get("resume_text", "").strip()
        if not resume_text:
            message = {"status": "error", "msg": "请先粘贴简历内容"}
        else:
            text_lower = resume_text.lower()
            # 用技能词典做一个简单的匹配（以后你们也可以换成 LLM 抽取）
            extracted = []
            for skill in SKILL_LIST:
                if skill.lower() in text_lower:
                    extracted.append(skill)
            extracted_skills = filter_none_skills(extracted)

            if not extracted_skills:
                message = {
                    "status": "error",
                    "msg": "未在简历中识别出技能，可以检查技能词典或简历内容"
                }
            else:
                ok = create_user_skill_relation(user_id, extracted_skills)
                if ok:
                    message = {
                        "status": "success",
                        "msg": f"已从简历中识别 {len(extracted_skills)} 个技能，并写入知识图谱"
                    }
                else:
                    message = {"status": "error", "msg": "写入知识图谱失败，请稍后重试"}

    # 从图谱里查当前这个用户已经挂上的技能
    try:
        with neo4j_driver.session() as session:
            recs = session.run("""
                MATCH (p:Person {id: $user_id})-[:HAS_SKILL]->(s:Skill)
                RETURN s.name AS name
                ORDER BY name
            """, user_id=user_id).data()
        graph_skills = filter_none_skills([r["name"] for r in recs])
    except Exception as e:
        print("[ERROR] resume_kg_page:", e)

    return render_template(
        "resume_kg.html",
        resume_text=resume_text,
        extracted_skills=extracted_skills,
        graph_skills=graph_skills,
        message=message,
        skill_list=SKILL_LIST,
    )


@app.route("/match-diag", methods=["GET", "POST"])
def match_diag_page():
    """岗位匹配与技能诊断（集成手动技能输入）"""
    # 获取所有岗位
    try:
        with neo4j_driver.session() as session:
            jobs = session.run("MATCH (j:Job) RETURN j.job_id AS id, j.name AS name").data()
        all_jobs = jobs
    except Exception as e:
        print(f"[ERROR] 查询岗位失败: {e}")
        all_jobs = []

    # 初始化变量
    match_result = None
    skill_submit_msg = None
    form = SkillForm()
    user_id = "manual_input_user"
    user_existing_skills = []
    llm_report = None  # 初始化LLM报告
    radar_data = None  # 雷达图数据

    # GET请求加载用户技能
    if request.method == "GET":
        try:
            with neo4j_driver.session() as session:
                recs = session.run("""
                    MATCH (p:Person {id: $user_id})-[:HAS_SKILL]->(s:Skill)
                    RETURN s.name AS name
                    ORDER BY name
                """, user_id=user_id).data()
            user_existing_skills = filter_none_skills([r["name"] for r in recs])
        except Exception as e:
            print(f"[ERROR] 加载用户已有技能失败: {e}")
            user_existing_skills = []

    if request.method == "POST":
        # 处理技能提交
        if 'skills' in request.form:
            try:
                import json
                skills = json.loads(request.form['skills'])
                skills = filter_none_skills(skills)  # 过滤None值
                if not skills:
                    skill_submit_msg = {"status": "error", "msg": "请至少添加一个技能"}
                else:
                    success = create_user_skill_relation(user_id, skills)
                    if success:
                        skill_submit_msg = {"status": "success", "msg": f"技能提交成功！已添加{len(skills)}个技能"}
                    else:
                        skill_submit_msg = {"status": "error", "msg": "技能提交失败，请重试"}
            except Exception as e:
                skill_submit_msg = {"status": "error", "msg": f"提交失败：{str(e)}"}

        # 处理岗位匹配
        if 'target_job_id' in request.form:
            target_job_id = request.form.get("target_job_id", "").strip()
            if not target_job_id:
                match_result = {"error": "请选择目标岗位"}
            else:
                try:
                    # 查询用户技能
                    with neo4j_driver.session() as session:
                        user_records = session.run("""
                            MATCH (p:Person {id: $user_id})-[:HAS_SKILL]->(s:Skill)
                            RETURN s.name AS name
                        """, user_id=user_id).data()
                    user_skills = filter_none_skills([r["name"] for r in user_records])

                    if not user_skills:
                        match_result = {"error": "请先提交个人技能"}
                    else:
                        # 查询岗位技能需求
                        with neo4j_driver.session() as session:
                            job_records = session.run("""
                                MATCH (j:Job {job_id: $job_id})-[r:REQUIRES]->(s:Skill)  
                                RETURN s.name AS name, coalesce(r.weight, 1) AS weight, j.name AS job_name  
                            """, job_id=target_job_id).data()

                        if not job_records:
                            match_result = {"error": f"岗位无技能需求数据"}
                        else:
                            # 过滤掉技能名称为None的记录
                            job_records = [r for r in job_records if r["name"] is not None and r["name"].strip() != ""]

                            if not job_records:
                                match_result = {"error": f"岗位技能数据异常，请检查数据库"}
                            else:
                                # 计算匹配度
                                req_dict = {r["name"]: r["weight"] for r in job_records}
                                owned = [s for s in req_dict if s in user_skills]
                                missing = [s for s in req_dict if s not in user_skills]
                                total_w = sum(req_dict.values())
                                owned_w = sum(req_dict[s] for s in owned)
                                score = round((owned_w / total_w) * 100) if total_w > 0 else 0

                                missing_sorted = sorted(missing, key=lambda x: req_dict[x], reverse=True)
                                recommend = missing_sorted[:3]

                                # 重新加载用户技能（用于页面显示）
                                try:
                                    with neo4j_driver.session() as session:
                                        recs = session.run("""
                                            MATCH (p:Person {id: $user_id})-[:HAS_SKILL]->(s:Skill)
                                            RETURN s.name AS name
                                            ORDER BY name
                                        """, user_id=user_id).data()
                                    user_existing_skills = filter_none_skills([r["name"] for r in recs])
                                except Exception as e:
                                    print(f"[ERROR] 重新加载用户技能失败: {e}")

                                # 雷达图数据
                                skill_dimensions = list(req_dict.keys())
                                if skill_dimensions:  # 确保有技能维度
                                    max_weight = max(req_dict.values()) if req_dict else 1
                                    job_weights = [round((req_dict[skill] / max_weight) * 10, 1) for skill in
                                                   skill_dimensions]
                                    user_weights = [
                                        round((req_dict[skill] / max_weight) * 10, 1) if skill in user_skills else 0 for
                                        skill in skill_dimensions]
                                    radar_data = {
                                        "dimensions": skill_dimensions,
                                        "job_weights": job_weights,
                                        "user_weights": user_weights
                                    }
                                else:
                                    radar_data = None

                                # 构建匹配结果（含竞争力总结）
                                match_result = {
                                    "job_name": job_records[0]["job_name"],
                                    "match_score": score,
                                    "match_level": get_match_level(score),
                                    "owned_skills": owned,
                                    "missing_skills": missing,
                                    "recommend_skills": recommend,
                                    "radar_data": radar_data,
                                    "competition_summary": get_competition_summary(score, owned, missing)
                                }

                                # 生成LLM报告
                                llm_report = generate_llm_report(match_result, user_skills)
                except Exception as e:
                    error_msg = f"系统内部错误：{str(e)}"
                    print(f"[CRITICAL ERROR] {error_msg}")
                    import traceback
                    traceback.print_exc()
                    match_result = {"error": error_msg}
                    llm_report = None

    return render_template(
        "match_diag.html",
        all_jobs=all_jobs,
        form=form,
        skill_submit_msg=skill_submit_msg,
        match_result=match_result,
        skill_list=SKILL_LIST,
        llm_report=llm_report,
        user_existing_skills=user_existing_skills,
        radar_data=radar_data if match_result and 'error' not in match_result else None
    )


@app.route("/path-reco", methods=["GET", "POST"])
def path_reco_page():
    """职业路径推荐"""

    target_job_id = request.args.get('job_id', '')  # 从URL获取目标岗位ID

    # 获取所有岗位
    try:
        with neo4j_driver.session() as session:
            jobs = session.run("""
                MATCH (j:Job) 
                RETURN j.job_id AS id, j.name AS name 
                ORDER BY j.job_id
            """).data()
        all_jobs = jobs
    except Exception as e:
        print(f"[ERROR] 查询岗位失败: {e}")
        all_jobs = []

    person_id = "manual_input_user"
    job_reco = []
    skill_path = None

    if request.method == "POST":
        target_job_id = request.form.get("target_job_id", "").strip()

    try:
        # 查询用户技能
        with neo4j_driver.session() as session:
            user_records = session.run("""
                MATCH (p:Person {id: $person_id})-[:HAS_SKILL]->(s:Skill)
                RETURN s.name AS name
            """, person_id=person_id).data()
        user_skills = filter_none_skills([r["name"] for r in user_records]) or ["Python", "SQL"]

        # 计算岗位匹配度推荐
        with neo4j_driver.session() as session:
            all_job_skills = session.run("""
                MATCH (j:Job)-[r:REQUIRES]->(s:Skill)
                RETURN j.job_id AS job_id, j.name AS job_name, j.city AS city, collect({name: s.name, weight: r.weight}) AS skill_list
            """).data()

        # 计算每个岗位的匹配度
        for rec in all_job_skills:
            # 过滤技能名称中的None值
            skill_list = [s for s in rec["skill_list"] if s["name"] is not None and s["name"].strip() != ""]
            skill_names = [s["name"] for s in skill_list]
            total_weight = sum([s["weight"] for s in skill_list])
            overlap_weight = sum([s["weight"] for s in skill_list if s["name"] in user_skills])
            rate = round((overlap_weight / total_weight) * 100) if total_weight > 0 else 0

            job_reco.append({
                "job_id": rec["job_id"],
                "job_name": rec["job_name"],
                "city": rec.get("city", "未知"),
                "match_rate": rate,
                "overlap_skills": list(set(user_skills) & set(skill_names))
            })

        # 排序取TOP5
        job_reco.sort(key=lambda x: x["match_rate"], reverse=True)
        job_reco = job_reco[:5]

        # 生成目标岗位技能路径
        if target_job_id:
            with neo4j_driver.session() as session:
                target_records = session.run("""
                    MATCH (j:Job {job_id: $job_id})-[r:REQUIRES]->(s:Skill)
                    RETURN s.name AS name, coalesce(r.weight, 1) AS weight, j.name AS job_name
                """, job_id=target_job_id).data()

            if not target_records:
                skill_path = {"error": f"岗位ID【{target_job_id}】无技能需求数据"}
            else:
                # 过滤掉None值
                target_records = [r for r in target_records if r["name"] is not None and r["name"].strip() != ""]
                if not target_records:
                    skill_path = {"error": f"岗位技能数据异常，请检查数据库"}
                else:
                    target_dict = {r["name"]: r["weight"] for r in target_records}
                    owned = [s for s in target_dict if s in user_skills]
                    missing = [s for s in target_dict if s not in user_skills]
                    missing_sorted = sorted(missing, key=lambda x: target_dict[x], reverse=True)

                    # 拆分学习阶段
                    phase1 = missing_sorted[:2] if len(missing_sorted) >= 2 else missing_sorted
                    phase2 = missing_sorted[2:4] if len(missing_sorted) >= 4 else (
                        missing_sorted[2:] if len(missing_sorted) > 2 else [])
                    phase3 = missing_sorted[4:] if len(missing_sorted) > 4 else []

                    # ========== 动态生成岗位适配的学习建议 ==========
                    job_name = target_records[0]["job_name"]
                    # 根据岗位类型判断学习建议
                    if "开发" in job_name:
                        phase1_action = "优先掌握基础语法与框架使用，建议通过官方文档+Demo项目练习（如：搭建简单接口）"
                        phase2_action = "进阶学习性能优化与中间件，结合开源项目（如：参与GitHub小项目）巩固"
                        phase3_action = "深入框架源码与分布式架构，尝试独立开发中型系统（如：用户管理平台）"
                    elif "分析" in job_name:
                        phase1_action = "优先掌握数据清洗与可视化工具，建议通过 Kaggle 入门项目练习（如：泰坦尼克号数据分析）"
                        phase2_action = "进阶学习统计模型与算法，结合企业数据集（如：电商用户行为分析）实践"
                        phase3_action = "深入机器学习与业务建模，参与真实业务场景的数据分析项目"
                    elif "研究" in job_name:
                        phase1_action = "优先掌握基础理论与工具，建议通过论文复现+小型实验练习（如：复现经典算法）"
                        phase2_action = "进阶学习前沿技术与实验设计，结合开源数据集（如：论文配套数据集）实践"
                        phase3_action = "深入领域前沿与创新研究，尝试发表论文或参与竞赛（如：Kaggle竞赛）"
                    else:
                        phase1_action = "优先掌握基础技能，建议通过视频教程+小项目练习"
                        phase2_action = "进阶技能学习，结合实战项目巩固"
                        phase3_action = "核心技能突破，参与真实业务场景项目"

                    # 构建图谱数据
                    nodes = []
                    # 已掌握技能
                    for skill in owned:
                        nodes.append({
                            "name": skill,
                            "category": 0,
                            "symbolSize": 60 + (target_dict[skill] * 5),
                            "itemStyle": {"color": "#10b981"},
                            "tooltip": f"已掌握技能：{skill}\n权重：{target_dict[skill]}"
                        })
                    # 待学技能分阶段
                    for skill in phase1:
                        nodes.append({
                            "name": skill, "category": 1, "symbolSize": 60 + (target_dict[skill] * 5),
                            "itemStyle": {"color": "#2563eb"},
                            "tooltip": f"阶段1学习：{skill}\n权重：{target_dict[skill]}"
                        })
                    for skill in phase2:
                        nodes.append({
                            "name": skill, "category": 2, "symbolSize": 60 + (target_dict[skill] * 5),
                            "itemStyle": {"color": "#f59e0b"},
                            "tooltip": f"阶段2学习：{skill}\n权重：{target_dict[skill]}"
                        })
                    for skill in phase3:
                        nodes.append({
                            "name": skill, "category": 3, "symbolSize": 60 + (target_dict[skill] * 5),
                            "itemStyle": {"color": "#ef4444"},
                            "tooltip": f"阶段3学习：{skill}\n权重：{target_dict[skill]}"
                        })
                    # 目标岗位节点
                    nodes.append({
                        "name": target_records[0]["job_name"], "category": 4, "symbolSize": 80,
                        "itemStyle": {"color": "#8b5cf6"}, "tooltip": f"目标岗位：{target_records[0]['job_name']}"
                    })

                    # 构建连线
                    links = []
                    # 已掌握→阶段1
                    for o_skill in owned:
                        for p1_skill in phase1:
                            links.append({"source": o_skill, "target": p1_skill, "lineStyle": {"width": 2}})
                    # 阶段1→阶段2
                    for p1_skill in phase1:
                        for p2_skill in phase2:
                            links.append({"source": p1_skill, "target": p2_skill, "lineStyle": {"width": 1.5}})
                    # 阶段2→阶段3
                    for p2_skill in phase2:
                        for p3_skill in phase3:
                            links.append({"source": p2_skill, "target": p3_skill, "lineStyle": {"width": 1}})
                    # 最后阶段→目标岗位
                    final_phase = phase3 if phase3 else (phase2 if phase2 else phase1)
                    for skill in final_phase:
                        links.append({
                            "source": skill, "target": target_records[0]["job_name"],
                            "lineStyle": {"width": 3, "color": "#2563eb"}
                        })

                    # 路径描述
                    if not missing_sorted:
                        desc = "✅ 你已具备该岗位的所有核心技能，可直接投递！"
                    elif len(missing_sorted) <= 2:
                        desc = f"⚠️ 优先学习（基础层）：{safe_join(missing_sorted[:2], '、')}，掌握后可达到岗位基础要求。"
                    elif len(missing_sorted) <= 4:
                        desc = f"""📚 分两阶段学习：
1. 第一阶段（1-2个月）：{safe_join(missing_sorted[:2], '、')}（核心权重技能）；
2. 第二阶段（2-3个月）：{safe_join(missing_sorted[2:], '、')}（辅助技能）。"""
                    else:
                        desc = f"""📚 分三阶段学习：
1. 第一阶段（1-2个月）：{safe_join(missing_sorted[:2], '、')}（核心权重技能）；
2. 第二阶段（2-3个月）：{safe_join(missing_sorted[2:4], '、')}（重要技能）；
3. 第三阶段（3-4个月）：{safe_join(missing_sorted[4:], '、')}（拓展技能）。"""

                    skill_path = {
                        "target_job_id": target_job_id,
                        "target_job_name": target_records[0]["job_name"],
                        "owned_skills": owned,
                        "missing_skills": missing_sorted,
                        "phase1": {
                            "skills": phase1,
                            "time_range": "1-2个月",
                            "action": "优先掌握基础技能，建议通过视频教程+小项目练习"
                        },
                        "phase2": {
                            "skills": phase2,
                            "time_range": "2-3个月",
                            "action": "进阶技能学习，结合实战项目巩固"
                        },
                        "phase3": {
                            "skills": phase3,
                            "time_range": "3-4个月",
                            "action": "核心技能突破，参与真实业务场景项目"
                        },
                        "path_desc": desc,
                        "graph_data": {"nodes": nodes, "links": links}
                    }

    except Exception as e:
        error_msg = f"系统内部错误：{str(e)}"
        print(f"[CRITICAL ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        job_reco = []
        skill_path = {"error": error_msg}

    return render_template(
        "path_reco.html",
        all_jobs=all_jobs,
        person_id=person_id,
        job_reco=job_reco,
        skill_path=skill_path,
        selected_job_id=target_job_id
    )


# ========== 启动程序 ==========
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)