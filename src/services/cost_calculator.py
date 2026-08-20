from decimal import Decimal, ROUND_HALF_UP
from src.config.config_parser import settings


class CostCalculator:
    @staticmethod
    def calculate(
        input_tokens: int,
        output_tokens: int,
    ) -> dict:
        input_cost = (
            Decimal(input_tokens)
            / Decimal(1_000_000)
            * Decimal(str(settings.input_price_per_million))
        )

        output_cost = (
            Decimal(output_tokens)
            / Decimal(1_000_000)
            * Decimal(str(settings.output_price_per_million))
        )

        total_cost = input_cost + output_cost

        return {
            "input_cost_usd": float(
                input_cost.quantize(
                    Decimal("0.00000001"),
                    rounding=ROUND_HALF_UP,
                )
            ),
            "output_cost_usd": float(
                output_cost.quantize(
                    Decimal("0.00000001"),
                    rounding=ROUND_HALF_UP,
                )
            ),
            "total_cost_usd": float(
                total_cost.quantize(
                    Decimal("0.00000001"),
                    rounding=ROUND_HALF_UP,
                )
            ),
        }
