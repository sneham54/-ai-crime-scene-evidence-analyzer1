import test from 'node:test'
import assert from 'node:assert/strict'
import { withLoadingState } from './loadingState.js'

test('loading state is cleared after successful work', async () => {
  const states = []
  const result = await withLoadingState(value => states.push(value), async () => 'done')
  assert.equal(result, 'done')
  assert.deepEqual(states, [true, false])
})

test('loading state is cleared when work rejects', async () => {
  const states = []
  await assert.rejects(
    withLoadingState(value => states.push(value), async () => { throw new Error('request failed') }),
    /request failed/
  )
  assert.deepEqual(states, [true, false])
})
