"""(...existing header...)"""
# [Keep all existing imports and code until _get_rag_context method]

    def _get_rag_context(self) -> RAGContext:
        """Get RAG context for pattern detection."""
        try:
            if RAGRetriever is None:
                return RAGContext()
            
            if self._rag_retriever is None:
                self._rag_retriever = RAGRetriever(use_multi_hyde=False)  # Can enable later
            
            # NEW API: build_council_context
            current_state = {
                'symbol': self.coin,
                'timeframe': self.timeframe,
                'direction': 'long',  # TODO: detect from features
                'features': self.features
            }
            
            context = self._rag_retriever.build_council_context(
                current_state=current_state,
                top_k=5
            )
            
            # context contains:
            # - similar_patterns
            # - historical_win_rate
            # - recommendation
            # - confidence
            
            return RAGContext(
                similar_patterns=context.get('similar_patterns', []),
                recent_trades=context.get('similar_patterns', []),  # Using patterns as proxy
                relevant_lessons=context.get('lessons', []),
                historical_win_rate=context.get('historical_win_rate', 0.5),
                avg_pnl=context.get('avg_pnl', 0),
                recommendation=context.get('recommendation', ''),
                confidence=context.get('confidence', 0.5),
            )
            
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return RAGContext()

# [Keep rest of the file unchanged]
