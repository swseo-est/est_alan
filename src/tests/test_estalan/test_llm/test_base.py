import pytest
from unittest.mock import Mock, AsyncMock, patch

from estalan.llm.base import AlanBaseChatModelWrapper


class TestAlanBaseChatModelWrapper:
    """AlanBaseChatModelWrapper 클래스의 함수별 단위 테스트"""

    @pytest.fixture
    def mock_model(self):
        """테스트용 Mock 모델 생성"""
        model = Mock()
        model.__class__.__name__ = "MockChatModel"
        return model

    @pytest.fixture
    def wrapper(self, mock_model):
        """테스트용 래퍼 인스턴스 생성"""
        return AlanBaseChatModelWrapper(mock_model)

    @pytest.fixture
    def wrapper_with_name(self, mock_model):
        """사용자 정의 이름을 가진 래퍼 인스턴스 생성"""
        return AlanBaseChatModelWrapper(mock_model, name="CustomName")

    def test_init_default_name(self, mock_model):
        """__init__ 함수 테스트: 기본 이름 사용"""
        wrapper = AlanBaseChatModelWrapper(mock_model)
        assert wrapper._model == mock_model
        assert wrapper.name == "MockChatModel"

    def test_init_custom_name(self, mock_model):
        """__init__ 함수 테스트: 사용자 정의 이름 사용"""
        wrapper = AlanBaseChatModelWrapper(mock_model, name="CustomName")
        assert wrapper._model == mock_model
        assert wrapper.name == "CustomName"

    def test_getattr_delegation(self, wrapper, mock_model):
        """__getattr__ 함수 테스트: 속성 위임"""
        # 모델에 존재하지 않는 속성에 접근할 때 위임
        mock_model.some_attribute = "test_value"
        assert wrapper.some_attribute == "test_value"

    def test_getattr_model_method(self, wrapper, mock_model):
        """__getattr__ 함수 테스트: 모델 메서드 위임"""
        # Mock 객체에 메서드를 추가
        mock_model.some_method = Mock(return_value="method_result")
        assert wrapper.some_method() == "method_result"
        mock_model.some_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_ainvoke_success(self, wrapper, mock_model):
        """ainvoke 함수 테스트: 성공 케이스"""
        expected_result = "test response"
        mock_model.ainvoke = AsyncMock(return_value=expected_result)
        
        result = await wrapper.ainvoke("test input")
        
        assert result == expected_result
        mock_model.ainvoke.assert_called_once_with("test input")

    @pytest.mark.asyncio
    async def test_ainvoke_with_retry(self, wrapper, mock_model):
        """ainvoke 함수 테스트: 재시도 로직"""
        expected_result = "test response"
        mock_model.ainvoke = AsyncMock()
        mock_model.ainvoke.side_effect = [
            Exception("First error"),
            Exception("Second error"),
            expected_result
        ]
        
        result = await wrapper.ainvoke("test input")
        
        assert result == expected_result
        assert mock_model.ainvoke.call_count == 3

    def test_invoke_success(self, wrapper, mock_model):
        """invoke 함수 테스트: 성공 케이스"""
        expected_result = "test response"
        mock_model.invoke.return_value = expected_result
        
        result = wrapper.invoke("test input")
        
        assert result == expected_result
        mock_model.invoke.assert_called_once_with("test input")

    def test_invoke_with_retry(self, wrapper, mock_model):
        """invoke 함수 테스트: 재시도 로직"""
        expected_result = "test response"
        mock_model.invoke.side_effect = [
            Exception("First error"),
            Exception("Second error"),
            expected_result
        ]
        
        result = wrapper.invoke("test input")
        
        assert result == expected_result
        assert mock_model.invoke.call_count == 3

    def test_stream_json(self, wrapper, mock_model):
        """stream_json 함수 테스트: JSON 스트리밍"""
        mock_chunks = ["chunk1", "chunk2", "chunk3"]
        mock_model.stream.return_value = mock_chunks
        
        result = list(wrapper.stream_json("test input"))
        
        expected = [
            {"model": "MockChatModel", "content": chunk}
            for chunk in mock_chunks
        ]
        assert result == expected
        mock_model.stream.assert_called_once_with("test input")

    def test_pre_hook_default_implementation(self, wrapper):
        """_pre_hook 함수 테스트: 기본 구현"""
        # 기본 구현은 아무것도 하지 않음
        result = wrapper._pre_hook("arg1", "arg2", kwarg1="value1")
        assert result is None

    def test_post_hook_default_implementation(self, wrapper):
        """_post_hook 함수 테스트: 기본 구현"""
        test_result = "test result"
        result = wrapper._post_hook(test_result)
        assert result == test_result

    def test_pre_hook_override(self):
        """_pre_hook 함수 테스트: 오버라이드"""
        class CustomWrapper(AlanBaseChatModelWrapper):
            def _pre_hook(self, *args, **kwargs):
                self.pre_hook_called = True
                self.pre_hook_args = args
                self.pre_hook_kwargs = kwargs
        
        mock_model = Mock()
        mock_model.__class__.__name__ = "MockChatModel"
        wrapper = CustomWrapper(mock_model)
        
        wrapper._pre_hook("arg1", kwarg1="value1")
        
        assert wrapper.pre_hook_called
        assert wrapper.pre_hook_args == ("arg1",)
        assert wrapper.pre_hook_kwargs == {"kwarg1": "value1"}

    def test_post_hook_override(self):
        """_post_hook 함수 테스트: 오버라이드"""
        class CustomWrapper(AlanBaseChatModelWrapper):
            def _post_hook(self, result):
                return f"processed: {result}"
        
        mock_model = Mock()
        mock_model.__class__.__name__ = "MockChatModel"
        wrapper = CustomWrapper(mock_model)
        
        result = wrapper._post_hook("test result")
        
        assert result == "processed: test result"

    def test_invoke_calls_pre_and_post_hooks(self, wrapper, mock_model):
        """invoke 함수 테스트: 훅 호출 확인"""
        expected_result = "test response"
        mock_model.invoke.return_value = expected_result
        
        # 훅을 오버라이드하여 호출 여부 확인
        pre_hook_called = False
        post_hook_called = False
        
        def custom_pre_hook(*args, **kwargs):
            nonlocal pre_hook_called
            pre_hook_called = True
        
        def custom_post_hook(result):
            nonlocal post_hook_called
            post_hook_called = True
            return result
        
        wrapper._pre_hook = custom_pre_hook
        wrapper._post_hook = custom_post_hook
        
        result = wrapper.invoke("test input")
        
        assert pre_hook_called
        assert post_hook_called
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_ainvoke_calls_pre_and_post_hooks(self, wrapper, mock_model):
        """ainvoke 함수 테스트: 훅 호출 확인"""
        expected_result = "test response"
        mock_model.ainvoke = AsyncMock(return_value=expected_result)
        
        # 훅을 오버라이드하여 호출 여부 확인
        pre_hook_called = False
        post_hook_called = False
        
        def custom_pre_hook(*args, **kwargs):
            nonlocal pre_hook_called
            pre_hook_called = True
        
        def custom_post_hook(result):
            nonlocal post_hook_called
            post_hook_called = True
            return result
        
        wrapper._pre_hook = custom_pre_hook
        wrapper._post_hook = custom_post_hook
        
        result = await wrapper.ainvoke("test input")
        
        assert pre_hook_called
        assert post_hook_called
        assert result == expected_result
