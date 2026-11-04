"""Correlate native Codex approval events; observations grant no authority."""
import json
from copy import deepcopy

from .action_control import ActionRefused


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value):
    if type(value) is not str or not value or len(value) > 256 or '\0' in value:
        raise ActionRefused('invalid_native_review_identity')
    return value


class NativeReviewBinding:
    def __init__(self, arguments, thread_id, turn_id):
        if type(arguments) is not dict:
            raise ActionRefused('native_review_arguments_required')
        self.arguments = canonical(arguments)
        self.thread_id, self.turn_id = identity(thread_id), identity(turn_id)
        self.item_id = None
        self.started = None
        self.completed = None
        self.failure = None

    @property
    def approved(self):
        return self.completed is not None and self.completed['review']['status'] == 'approved'

    @property
    def terminal(self):
        return deepcopy(self.completed['review']) if self.completed else None

    def observe(self, row):
        try:
            self._observe(row)
        except ActionRefused as exc:
            params=row.get('params',{}) if isinstance(row,dict) else {}
            item=params.get('item',{}) if isinstance(params,dict) else {}
            self.failure={'method':row.get('method') if isinstance(row,dict) else None,
                'error_type':type(exc).__name__,'reason':str(exc),
                'item_type':item.get('type') if isinstance(item,dict) else None,
                'arguments_match':canonical(item.get('arguments'))==self.arguments if isinstance(item,dict) else False,
                'argument_type':type(item.get('arguments')).__name__ if isinstance(item,dict) else None,
                'server_match':item.get('server')=='tare_action' if isinstance(item,dict) else False,
                'tool_match':item.get('tool')=='execute_exact_command' if isinstance(item,dict) else False,
                'already_bound':self.item_id is not None}
            raise
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            fields={'item','type','threadId','turnId','arguments','server','tool','id',
                    'action','targetItemId','reviewId','startedAtMs','completedAtMs','decisionSource','review','status'}
            key=exc.args[0] if isinstance(exc,KeyError) and exc.args else None
            self.failure={'method':row.get('method') if isinstance(row,dict) else None,
                          'error_type':type(exc).__name__,
                          'missing_field':key if isinstance(key,str) and key in fields else None}
            raise ActionRefused('malformed_native_review') from None

    def _observe(self, row):
        method, params = row.get('method'), row.get('params', {})
        if method == 'item/started':
            item = params['item']
            if item['type'] in ('userMessage', 'agentMessage', 'reasoning'):
                return
            if (item['type'] != 'mcpToolCall' or self.item_id is not None
                    or params['threadId'] != self.thread_id or params['turnId'] != self.turn_id
                    or item['server'] != 'tare_action' or item['tool'] != 'execute_exact_command'
                    or canonical(item['arguments']) != self.arguments):
                raise ActionRefused('native_review_tool_mismatch')
            self.item_id = identity(item['id'])
            return
        if method not in ('item/autoApprovalReview/started', 'item/autoApprovalReview/completed'):
            return
        action = params['action']
        if (params['threadId'] != self.thread_id or params['turnId'] != self.turn_id
                or self.item_id is None or params['targetItemId'] != self.item_id
                or action.get('type') != 'mcpToolCall' or action.get('server') != 'tare_action'
                or action.get('toolName') != 'execute_exact_command'
                or set(action) - {'type', 'server', 'toolName', 'connectorId', 'connectorName', 'toolTitle'}
                or any(action.get(key) is not None for key in ('connectorId', 'connectorName', 'toolTitle'))):
            raise ActionRefused('native_review_subject_mismatch')
        identity(params['reviewId'])
        at = params['startedAtMs']
        if type(at) is not int or at < 0:
            raise ActionRefused('native_review_time_invalid')
        if method.endswith('/started'):
            if self.started is not None or self.completed is not None or params['review']['status'] != 'inProgress':
                raise ActionRefused('native_review_start_invalid')
            self.started = deepcopy(params)
            return
        if self.started is None or self.completed is not None:
            raise ActionRefused('native_review_completion_invalid')
        for key in ('reviewId', 'startedAtMs', 'action', 'targetItemId', 'threadId', 'turnId'):
            if canonical(params[key]) != canonical(self.started[key]):
                raise ActionRefused('native_review_completion_mismatch')
        end = params['completedAtMs']
        if (type(end) is not int or end < at or params['decisionSource'] != 'agent'
                or params['review']['status'] not in ('approved', 'denied', 'timedOut', 'aborted')):
            raise ActionRefused('native_review_terminal_invalid')
        self.completed = deepcopy(params)
