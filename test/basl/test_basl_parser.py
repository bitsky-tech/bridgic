import re
import pytest
import traceback

from test.basl.basl_parser import BASLparser

# test case 是一些合法的 DSL 代码
test_case = [
    ("""<graph>
            <RagInLawDomain 
                user_input={{user_input}}
                output_key="chunks"
                is_start="true"
            />
            <AnswerLawQuestion 
                user_input={{user_input}}
                law_chunks={{chunks}}
                is_output="true"
                dependencies=["RagInLawDomain"]
            />
        </graph>""", True, 3),
    ("""<sequential>
            <RagInFinanceDomain 
                user_input={{user_input}}
                output_key="chunks"
            />
            <AnswerFinanceQuestion 
                user_input={{user_input}}
                finance_chunks={{chunks}}
            />
        </sequential>""", True, 3),
    ("""<graph>
            <DomainClassifier
                user_input={{input}}
                automa={{self}}
                is_start="true"
            />
            <ChatBotInLaw
                user_input={{input}}
                is_output="true"
            />
            <ChatBotInFinance
                user_input={{input}}
                is_output="true"
            />
        </graph>""", True, 4),
    ("""<graph>
            <Worker0
                arg0={{user_input}}
                output_key="output0"
                is_start="true"
            />
            <concurrent
                id="WorkersBundle"
                output_key="output_list"
                output_type="dict/list"
                dependencies=["Worker0"]
            >
                <Worker1
                    arg1={{output0}}
                    output_key="output1"
                />
                <Worker2
                    arg2={{output0}}
                    output_key="output2"
                />
                <Worker3
                    arg3={{output0}}
                    output_key="output3"
                />
            </concurrent>
            <Worker4
                x1={{WorkersBundle.output[0]}}
                x2={{output_list[1]}}
                x3={{output_list[2]}}
                is_output="true"
                dependencies=["WorkersBundle"]
            />
        </graph> """, True, 7),
        ("""<graph>
            <Worker0
                arg0={{user_input}}
                output_key="output0"
                is_start="true"
            />
            <Worker1
                arg1={{output0}}
                dependencies=["Worker0"]
            />
            <Worker2
                id="MyWorker2"
                arg2={{output0}}
                dependencies=["Worker0"]
            />
            <Worker3
                arg3={{output0}}
                dependencies=["Worker0"]
            />
            <Worker4
                x1={{Worker1.output}}
                x2={{MyWorker2.output}}
                x3={{Worker3.output}}
                is_output="true"
                dependencies=["Worker1", "Worker2", "Worker3"]
            />
       </graph>""", True, 7),
       ("""<sequential>
            <Worker1 
                arg1={{user_input}}
                output_key="output1"
            />
            <Worker2
                arg2={{output1}}
                output_key="output2"
            />
            <Worker3
                automa={{self}}
                arg3={{output2}}
                ferry_to="Worker2"
            />
        </sequential>""", True, 4),
    ("""<graph>
            <DivideSearchSubtask
                user_input={{user_input}}
                output_key="subtasks"
                is_start="true"
            />
            <concurrent
                id="ConcurrentSearch"
                output_key="search_results"
                dependencies=["DivideSearchSubtask"]
            >
                {{
                    [<SearchBySubtask
                        id="SearchBySubtask_{{i}}"
                        sub_task={{t}}
                        /> 
                        for i,t in enumerate(subtasks)
                    ]
                }}
            </concurrent>
            <SynthesizeSearchResults
                search_results={{search_results}}
                is_output="true"
                dependencies=[ "ConcurrentSearch"]
            />
        </graph>""", True, 3),
        ("""
        <graph>
            <Worker0 arg0={{user_input}} output_key="output0" is_start="true"/>  
            <concurrent id="WorkersBundle" output_key="output_list" output_type="dict/list" dependencies=["Worker0"]>
                <Worker1 arg1={{output0}} output_key="output1"/>
                <Worker2 arg2={{output0}} output_key="output2"/>
                <Worker3 arg3={{output0}} output_key="output3"/>
            </concurrent>
            <Worker4 x1={{WorkersBundle.output[0]}} x2={{output_list[1]}} x3={{output_list[2]}} is_output="true" dependencies=["WorkersBundle"]/>
        </graph>
        """, True, 7),
        ("""
        <graph>
            <DivideGameProgrammingTask
                game_requirement={{user_input}}
                output_key="subtasks"
                is_start="true"
            />
            <sequential
                id="SequentialGenerateGameModule"
                bindle_mode="true"
                output_key="module_list"
                dependencies=["DivideGameProgrammingTask"]
            >
            {{
                [<ProgrammingGameModule 
                    user_requirement={{user_input}}
                    task={{t}} 
                    output_key={{f"game_module_{i}"}}
                    generated_modules={{
                        [module_list[j] for j in range(i)]
                    }}
                    /> 
                    for i,t in enumerate(subtasks)
                ]
            }}
            </sequential>
            <CombineWholeGame
                user_requirement={{user_input}}
                game_modules={{module_list}}
                is_output="true"
                dependencies=["SequentialGenerateGameModule"]
            />
      </graph>

        """, True, 4),
        ("""<graph>
            <RelevantToolsFinder
                user_input={{{"a": 1}}}
                output_key="relevant_tools"
                is_start="true"
            />
            <ToolCallingContextAssembler
                user_input={{user_input}}
                candidate_tools={{relevant_tools}}
                tool_results={{tool_results}}
                output_key="llm_context"
                dependencies=["RelevantToolsFinder"]
            />
            <LLMWorker
                prompt={{llm_context}}
                json_mode={{True}}
                output_key="completion_in_json"
                dependencies=["ToolCallingContextAssembler"]
            />
            <OneStepDecisionMaker
                automa={{self}}
                llm_output={{completion_in_json}}
                dependencies=["LLMWorker"]
            />
            <concurrent
                id="ConcurrentToolsCaller"
                tool_calls={{tool_calls}}
                output_key="tool_results"
                ferry_to="ToolCallingContextAssembler"
            >
            {{
                [<ToolWorker
                    id=f"Tool_{{t_c.id}}"
                    tool_call={{t_c}}
                    /> 
                    for t_c in tool_calls
                ]
            }}
            </concurrent>
            <OutputSummarizer
                is_output="true"
            />
      </graph>
        """, True, 10),
]

####################################################################################################################################################
# 测试 BASL 解释器
####################################################################################################################################################


@pytest.mark.parametrize("input, expected, num_of_python_expr_token", test_case)
def test_dsl_parser(input, expected, num_of_python_expr_token):
    """
    测试 BASL 解释器正确的把合法的 DSL 代码中的 {{ ... }} 代码块替换为 PYTHONEXPRTOKEN
    """
    bridgic_parser = BASLparser()

    input_after_pre_process = bridgic_parser._pre_process(input)
    count = input_after_pre_process.count("PYTHONEXPRTOKEN")
    assert count == num_of_python_expr_token


@pytest.mark.parametrize("input, expected, num_of_python_expr_token", test_case)
def test_dsl_parser_parser(input, expected, num_of_python_expr_token):
    """
    测试 BASL 解释器正确的解析合法的 DSL 代码
    """
    bridgic_parser = BASLparser()
    
    input_after_pre_process = bridgic_parser._pre_process(input)
    flag = None
    try:
        bridgic_parser._parser(input_after_pre_process)
    except Exception as e:
        traceback.print_exc()
        flag = False
    else:
        flag = True
    assert flag == expected

