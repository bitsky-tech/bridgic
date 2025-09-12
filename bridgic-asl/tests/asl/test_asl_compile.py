from bridgic.core.automa.worker import Worker
from bridgic.asl.compiler import ASLEntry
import pytest

from tests.asl.external_worker import (
    double_worker,
    async_double_worker,
    DoubleWorker,
)


def def_custom_worker1(automa, user_input: int):
    return user_input + 1


async def def_async_custom_worker1(automa, user_input: int):
    return user_input + 1


class CustomWorker1(Worker):
    async def arun(self, user_input: int):
        return user_input + 1


test_case = {
    "test_graph_automa_with_def_async_class": [
        """<graph>
            <def_custom_worker1 
                user_input={{user_input}} 
                output_key="work1_output" 
                is_start="true"
            />
            <def_async_custom_worker1 
                work1_res={{work1_output}} 
                output_key="work2_output" 
                dependencies=["def_custom_worker1"]
            />
            <CustomWorker1 
                work2_res={{work2_output}} 
                output_key="work3_output" 
                dependencies=["def_async_custom_worker1"]
                is_output="true"
            />
        </graph>
    """, 0, 3],
    "test_corroct_parse_import_worker": [
        """<graph>
            <def_custom_worker1 
                user_input={{user_input}} 
                output_key="work1_output" 
                is_start="true"
            />
            <double_worker 
                work1_res={{work1_output}} 
                output_key="work2_output" 
                dependencies=["def_custom_worker1"]
            />
            <async_double_worker 
                work2_res={{work2_output}} 
                output_key="work3_output" 
                dependencies=["double_worker"]
                is_output="true"
            />
        </graph>
    """, 0, 4]
}


####################################################################################################################################################
# test ASL compile
####################################################################################################################################################


def asl_entry_code_wraaper(asl_code: str, user_input: str):
    @ASLEntry()
    def asl_entry(user_input: str):
        return asl_code
    return asl_entry(user_input)


@pytest.mark.parametrize("asl_code, user_input, expected", test_case.values(), ids=test_case.keys())
def test_sample_asl(asl_code, user_input, expected):
    asl_res = asl_entry_code_wraaper(asl_code, user_input)
    assert asl_res == expected






