export async function withLoadingState(setLoading, task) {
  setLoading(true)
  try {
    return await task()
  } finally {
    setLoading(false)
  }
}
